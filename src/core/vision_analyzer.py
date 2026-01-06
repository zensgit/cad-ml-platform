"""
视觉分析模块 - 为CAD ML Platform添加识图能力
支持多种视觉AI服务的集成
"""

import base64
import io
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VisionProvider(Enum):
    """视觉服务提供商"""

    OPENAI = "openai"  # GPT-4 Vision
    ANTHROPIC = "anthropic"  # Claude Vision API
    AZURE = "azure"  # Azure Computer Vision
    GOOGLE = "google"  # Google Cloud Vision
    LOCAL = "local"  # 本地模型


@dataclass
class VisionResult:
    """视觉分析结果"""

    description: str  # 图像描述
    objects: List[Dict[str, Any]]  # 检测到的对象
    text: Optional[str]  # OCR文本
    drawings: Optional[Dict]  # CAD图纸元素
    dimensions: Optional[Dict]  # 尺寸标注
    confidence: float  # 置信度
    metadata: Dict[str, Any]  # 元数据


class VisionAnalyzer:
    """
    统一的视觉分析接口
    支持多种AI视觉服务
    """

    def __init__(self, provider: VisionProvider = VisionProvider.OPENAI):
        self.provider = provider
        self.clients = self._initialize_clients()

    def _initialize_clients(self) -> Dict:
        """初始化各种视觉服务客户端"""
        clients = {}

        # OpenAI GPT-4 Vision
        try:
            from openai import OpenAI

            clients["openai"] = OpenAI()
            logger.info("OpenAI Vision client initialized")
        except ImportError:
            logger.warning("OpenAI client not available")

        # Anthropic Claude Vision
        try:
            import anthropic

            clients["anthropic"] = anthropic.Anthropic()
            logger.info("Anthropic Vision client initialized")
        except ImportError:
            logger.warning("Anthropic client not available")

        # Azure Computer Vision
        try:
            from azure.cognitiveservices.vision.computervision import (  # noqa: F401
                ComputerVisionClient,
            )
            from msrest.authentication import CognitiveServicesCredentials  # noqa: F401

            # 需要配置Azure凭证
            clients["azure"] = None  # ComputerVisionClient配置
        except ImportError:
            logger.warning("Azure Vision client not available")

        # Google Cloud Vision
        try:
            from google.cloud import vision

            clients["google"] = vision.ImageAnnotatorClient()
        except ImportError:
            logger.warning("Google Vision client not available")

        # 本地模型
        try:
            import transformers  # noqa: F401
            from transformers import pipeline

            clients["local"] = pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-large"
            )
        except ImportError:
            logger.warning("Local vision model not available")

        return clients

    async def analyze_image(
        self,
        image: Union[
            str, bytes, Image.Image
        ],  # noqa: F401 (Image type retained for future processing)
        task: str = "general",
        options: Optional[Dict] = None,
    ) -> VisionResult:
        """
        分析图像

        Args:
            image: 图像路径、字节数据或PIL图像
            task: 任务类型 (general, cad, ocr, technical)
            options: 额外选项

        Returns:
            VisionResult: 分析结果
        """
        # 预处理图像
        image_data = self._preprocess_image(image)

        # 根据任务类型选择分析策略
        if task == "cad":
            return await self._analyze_cad_drawing(image_data, options)
        elif task == "ocr":
            return await self._perform_ocr(image_data, options)
        elif task == "technical":
            return await self._analyze_technical_drawing(image_data, options)
        else:
            return await self._general_analysis(image_data, options)

    async def _analyze_cad_drawing(
        self, image_data: bytes, options: Optional[Dict] = None
    ) -> VisionResult:
        """分析CAD图纸"""

        if self.provider == VisionProvider.OPENAI:
            return await self._openai_cad_analysis(image_data, options)
        elif self.provider == VisionProvider.ANTHROPIC:
            return await self._anthropic_cad_analysis(image_data, options)
        else:
            return await self._local_cad_analysis(image_data, options)

    async def _openai_cad_analysis(
        self, image_data: bytes, options: Optional[Dict] = None
    ) -> VisionResult:
        """使用OpenAI GPT-4 Vision分析CAD图纸"""

        client = self.clients.get("openai")
        if not client:
            raise ValueError("OpenAI client not initialized")

        # 编码图像
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # 构建提示词
        prompt = """
        请分析这张CAD技术图纸，识别并描述：
        1. 图纸类型（装配图、零件图、剖面图等）
        2. 主要零件和组件
        3. 尺寸标注和公差
        4. 材料和表面处理说明
        5. 技术要求和注释
        6. 图纸中的符号和标记

        请以JSON格式返回结果。
        """

        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        # 解析响应
        result_text = response.choices[0].message.content

        return VisionResult(
            description=result_text,
            objects=self._extract_objects(result_text),
            text=self._extract_text(result_text),
            drawings=self._extract_cad_elements(result_text),
            dimensions=self._extract_dimensions(result_text),
            confidence=0.95,
            metadata={"provider": "openai", "model": "gpt-4-vision"},
        )

    async def _anthropic_cad_analysis(
        self, image_data: bytes, options: Optional[Dict] = None
    ) -> VisionResult:
        """使用Claude Vision API分析CAD图纸"""

        client = self.clients.get("anthropic")
        if not client:
            raise ValueError("Anthropic client not initialized")

        # 编码图像
        base64_image = base64.b64encode(image_data).decode("utf-8")

        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": "分析这张CAD图纸，识别零件类型、尺寸、材料等信息。"},
                    ],
                }
            ],
        )

        result_text = message.content

        return VisionResult(
            description=result_text,
            objects=[],
            text=None,
            drawings=None,
            dimensions=None,
            confidence=0.95,
            metadata={"provider": "anthropic", "model": "claude-3"},
        )

    async def _local_cad_analysis(
        self, image_data: bytes, options: Optional[Dict] = None
    ) -> VisionResult:
        """使用本地模型分析（YOLO + OCR + 自定义模型）"""

        # 将字节转换为PIL图像
        image = Image.open(io.BytesIO(image_data))

        # 1. 使用YOLO检测对象
        objects = await self._detect_objects(image)

        # 2. 使用OCR提取文字
        text = await self._extract_text_ocr(image)

        # 3. 使用自定义CNN分析图纸特征
        features = await self._extract_cad_features(image)

        # 4. 综合分析
        description = self._generate_description(objects, text, features)

        return VisionResult(
            description=description,
            objects=objects,
            text=text,
            drawings=features.get("drawings"),
            dimensions=features.get("dimensions"),
            confidence=0.85,
            metadata={"provider": "local", "models": ["yolo", "tesseract", "custom"]},
        )

    async def _detect_objects(self, image: Image.Image) -> List[Dict]:
        """使用YOLO或其他模型检测对象"""
        try:
            import cv2  # noqa: F401
            from ultralytics import YOLO

            # 加载预训练的YOLO模型
            model = YOLO("yolov8n.pt")

            # 转换图像格式
            img_array = np.array(image)

            # 执行检测
            results = model(img_array)

            objects = []
            for r in results:
                for box in r.boxes:
                    objects.append(
                        {
                            "class": r.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy.tolist()[0],
                        }
                    )

            return objects

        except ImportError:
            logger.warning("YOLO not available, skipping object detection")
            return []

    async def _extract_text_ocr(self, image: Image.Image) -> str:
        """使用OCR提取文字"""
        try:
            import pytesseract  # noqa: F401

            # 执行OCR
            text = pytesseract.image_to_string(image, lang="eng+chi_sim")
            return text

        except ImportError:
            logger.warning("Tesseract not available, skipping OCR")
            return ""

    async def _extract_cad_features(self, image: Image.Image) -> Dict:
        """提取CAD特征（轻量级图像启发式）"""
        # 这里使用轻量级图像启发式来补足基础信息（避免依赖重型模型）
        features = {
            "drawings": {"lines": [], "circles": [], "arcs": [], "dimensions": []},
            "dimensions": {"overall_width": None, "overall_height": None, "tolerances": []},
        }

        try:
            gray = image.convert("L")
        except Exception:
            return features

        orig_width, orig_height = gray.size
        features["dimensions"]["overall_width"] = orig_width
        features["dimensions"]["overall_height"] = orig_height

        max_dim = 256
        if max(gray.size) > max_dim:
            scale = max_dim / max(gray.size)
            resized = (max(1, int(gray.size[0] * scale)), max(1, int(gray.size[1] * scale)))
            gray = gray.resize(resized, Image.BILINEAR)

        arr = np.array(gray)
        if arr.size == 0:
            return features

        # Simple binary mask for ink pixels (CAD lines tend to be darker)
        threshold = 200
        ink = arr < threshold
        total_pixels = ink.size
        ink_pixels = int(ink.sum())
        ink_ratio = ink_pixels / total_pixels if total_pixels else 0.0

        height, width = ink.shape
        visited = np.zeros_like(ink, dtype=bool)
        lines: List[Dict[str, Any]] = []
        circles: List[Dict[str, Any]] = []

        def _push_neighbors(stack, r, c):
            if r > 0 and ink[r - 1, c] and not visited[r - 1, c]:
                visited[r - 1, c] = True
                stack.append((r - 1, c))
            if r + 1 < height and ink[r + 1, c] and not visited[r + 1, c]:
                visited[r + 1, c] = True
                stack.append((r + 1, c))
            if c > 0 and ink[r, c - 1] and not visited[r, c - 1]:
                visited[r, c - 1] = True
                stack.append((r, c - 1))
            if c + 1 < width and ink[r, c + 1] and not visited[r, c + 1]:
                visited[r, c + 1] = True
                stack.append((r, c + 1))

        for r in range(height):
            for c in range(width):
                if not ink[r, c] or visited[r, c]:
                    continue
                visited[r, c] = True
                stack = [(r, c)]
                min_r = max_r = r
                min_c = max_c = c
                area = 0

                while stack:
                    cr, cc = stack.pop()
                    area += 1
                    if cr < min_r:
                        min_r = cr
                    if cr > max_r:
                        max_r = cr
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc
                    _push_neighbors(stack, cr, cc)

                if area < 12:
                    continue

                comp_width = max_c - min_c + 1
                comp_height = max_r - min_r + 1
                if comp_width <= 0 or comp_height <= 0:
                    continue
                aspect = max(comp_width, comp_height) / max(1, min(comp_width, comp_height))
                fill_ratio = area / float(comp_width * comp_height)

                if aspect >= 4.0:
                    lines.append(
                        {
                            "bbox": [int(min_c), int(min_r), int(max_c), int(max_r)],
                            "length": float(max(comp_width, comp_height)),
                            "fill_ratio": round(fill_ratio, 4),
                        }
                    )
                elif aspect <= 1.3 and fill_ratio >= 0.3:
                    radius = (comp_width + comp_height) / 4.0
                    circles.append(
                        {
                            "bbox": [int(min_c), int(min_r), int(max_c), int(max_r)],
                            "radius": round(radius, 2),
                            "fill_ratio": round(fill_ratio, 4),
                        }
                    )

        features["drawings"]["lines"] = lines
        features["drawings"]["circles"] = circles
        features["stats"] = {
            "ink_ratio": round(ink_ratio, 4),
            "components": int(len(lines) + len(circles)),
        }

        return features

    def _preprocess_image(self, image: Union[str, bytes, Image.Image]) -> bytes:
        """预处理图像为字节格式"""
        if isinstance(image, str):
            # 文件路径
            with open(image, "rb") as f:
                return f.read()
        elif isinstance(image, bytes):
            return image
        elif isinstance(image, Image.Image):
            # PIL图像
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _extract_objects(self, text: str) -> List[Dict]:
        """从文本中提取对象信息"""
        payload = self._extract_json_payload(text)
        objects = self._extract_objects_from_payload(payload)
        if objects:
            return objects
        return self._extract_objects_from_text(text)

    def _extract_text(self, text: str) -> str:
        """提取OCR文本"""
        payload = self._extract_json_payload(text)
        extracted = self._extract_text_from_payload(payload)
        if extracted:
            return extracted
        match = re.search(r"(?:text|ocr)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _extract_cad_elements(self, text: str) -> Dict:
        """提取CAD元素"""
        payload = self._extract_json_payload(text)
        elements = self._extract_cad_elements_from_payload(payload)
        return elements or {}

    def _extract_dimensions(self, text: str) -> Dict:
        """提取尺寸信息"""
        payload = self._extract_json_payload(text)
        dimensions = self._extract_dimensions_from_payload(payload)
        if dimensions:
            return dimensions
        values = []
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|cm|in|inch)\b", text, re.IGNORECASE):
            values.append({"value": float(match.group(1)), "unit": match.group(2).lower()})
        for match in re.finditer(r"[Ø∅]\s*(\d+(?:\.\d+)?)", text):
            values.append({"value": float(match.group(1)), "unit": None, "type": "diameter"})
        tolerances = []
        for match in re.findall(
            r"\+/-\s*(\d+(?:\.\d+)?)(?:\s*(mm|cm|in|inch))?",
            text,
            re.IGNORECASE,
        ):
            value, unit = match
            tolerances.append(
                {"type": "plus_minus", "value": float(value), "unit": unit.lower() if unit else None}
            )
        for match in re.findall(
            r"[±]\s*(\d+(?:\.\d+)?)(?:\s*(mm|cm|in|inch))?",
            text,
            re.IGNORECASE,
        ):
            value, unit = match
            tolerances.append(
                {"type": "plus_minus", "value": float(value), "unit": unit.lower() if unit else None}
            )
        for match in re.finditer(
            r"\+(\d+(?:\.\d+)?)(?:\s*(mm|cm|in|inch))?\s*/\s*-(\d+(?:\.\d+)?)(?:\s*(mm|cm|in|inch))?",
            text,
            re.IGNORECASE,
        ):
            plus_value, plus_unit, minus_value, minus_unit = match.groups()
            unit = None
            if plus_unit and minus_unit:
                if plus_unit.lower() == minus_unit.lower():
                    unit = plus_unit.lower()
            elif plus_unit:
                unit = plus_unit.lower()
            elif minus_unit:
                unit = minus_unit.lower()
            tolerances.append(
                {
                    "type": "asymmetric",
                    "plus": float(plus_value),
                    "minus": float(minus_value),
                    "unit": unit,
                }
            )
        if values or tolerances:
            return {"values": values, "tolerances": tolerances}
        return {}

    def _extract_json_payload(self, text: str) -> Optional[Any]:
        if not text:
            return None
        candidate = text.strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass
        for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
            block = match.group(1).strip()
            if not block:
                continue
            try:
                return json.loads(block)
            except Exception:
                continue
        snippet = self._find_json_snippet(text)
        if snippet:
            try:
                return json.loads(snippet)
            except Exception:
                pass
        return None

    def _find_json_snippet(self, text: str) -> Optional[str]:
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            snippet = self._match_brackets(text, idx)
            if snippet:
                return snippet
        return None

    def _match_brackets(self, text: str, start: int) -> Optional[str]:
        opener = text[start]
        if opener not in "{[":
            return None
        stack = [opener]
        in_string = False
        escape = False
        for idx in range(start + 1, len(text)):
            ch = text[idx]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                last = stack.pop()
                if (last == "{" and ch != "}") or (last == "[" and ch != "]"):
                    return None
                if not stack:
                    return text[start : idx + 1]
        return None

    def _extract_objects_from_payload(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return self._normalize_objects(payload)
        if isinstance(payload, dict):
            for key in (
                "objects",
                "components",
                "parts",
                "entities",
                "symbols",
                "annotations",
            ):
                value = payload.get(key)
                if value is not None:
                    return self._normalize_objects(value)
            for key in ("result", "results", "data"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    extracted = self._extract_objects_from_payload(nested)
                    if extracted:
                        return extracted
            if any(k in payload for k in ("name", "type", "class", "label")):
                return [payload]
        return []

    def _normalize_objects(self, value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            normalized: List[Dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    normalized.append(item)
                elif isinstance(item, str):
                    normalized.append({"name": item})
                else:
                    normalized.append({"value": item})
            return normalized
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            return [{"name": value}]
        return []

    def _extract_objects_from_text(self, text: str) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []
        for line in text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            if candidate.startswith(("-", "*")):
                name = candidate.lstrip("-* ").strip()
                if name:
                    objects.append({"name": name})
            else:
                match = re.match(r"^\d+[\).\s]+(.+)$", candidate)
                if match:
                    name = match.group(1).strip()
                    if name:
                        objects.append({"name": name})
            if len(objects) >= 50:
                break
        return objects

    def _extract_text_from_payload(self, payload: Any) -> Optional[str]:
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            for item in payload:
                extracted = self._extract_text_from_payload(item)
                if extracted:
                    return extracted
            return None
        if isinstance(payload, dict):
            for key in ("text", "ocr_text", "recognized_text", "raw_text", "extracted_text"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    joined = " ".join([item for item in value if isinstance(item, str)])
                    if joined:
                        return joined
            if "ocr" in payload:
                return self._extract_text_from_payload(payload.get("ocr"))
        return None

    def _extract_cad_elements_from_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("cad_elements", "drawings", "elements", "geometry", "features"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
                if isinstance(value, list):
                    return {"items": self._normalize_objects(value)}
            for key in ("result", "results", "data"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    extracted = self._extract_cad_elements_from_payload(nested)
                    if extracted:
                        return extracted
        return None

    def _extract_dimensions_from_payload(self, payload: Any) -> Optional[Dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("dimensions", "dimension", "sizes", "tolerances"):
                value = payload.get(key)
                if value is None:
                    continue
                if isinstance(value, dict):
                    return value
                if isinstance(value, list):
                    return {"items": value}
                if isinstance(value, str):
                    return {"raw": value}
                if isinstance(value, (int, float)):
                    return {"value": value}
            for key in ("result", "results", "data", "drawings"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    extracted = self._extract_dimensions_from_payload(nested)
                    if extracted:
                        return extracted
        if isinstance(payload, list):
            for item in payload:
                extracted = self._extract_dimensions_from_payload(item)
                if extracted:
                    return extracted
        return None

    def _generate_description(self, objects: List[Dict], text: str, features: Dict) -> str:
        """生成综合描述"""
        description = "图像分析结果：\n"

        if objects:
            description += f"检测到{len(objects)}个对象\n"

        if text:
            description += f"识别到文字：{text[:100]}...\n"

        if features:
            description += "提取到CAD特征\n"

        return description


class CADImageProcessor:
    """
    CAD图像预处理器
    优化图像质量以提高识别准确率
    """

    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """增强图像质量"""
        from PIL import ImageEnhance, ImageFilter

        # 增强对比度
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # 增强锐度
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        # 去噪
        image = image.filter(ImageFilter.MedianFilter(size=3))

        return image

    @staticmethod
    def convert_to_binary(image: Image.Image) -> Image.Image:
        """转换为二值图像（用于线条检测）"""
        import cv2

        # 转换为灰度图
        gray = image.convert("L")

        # 转换为numpy数组
        img_array = np.array(gray)

        # 二值化
        _, binary = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)

        # 转回PIL图像
        return Image.fromarray(binary)

    @staticmethod
    def detect_edges(image: Image.Image) -> Image.Image:
        """边缘检测（用于提取线条）"""
        import cv2

        # 转换为numpy数组
        img_array = np.array(image.convert("L"))

        # Canny边缘检测
        edges = cv2.Canny(img_array, 50, 150)

        return Image.fromarray(edges)


# 使用示例
async def main():
    """使用示例"""

    # 初始化分析器
    analyzer = VisionAnalyzer(provider=VisionProvider.OPENAI)

    # 分析CAD图纸图像
    result = await analyzer.analyze_image("cad_drawing.jpg", task="cad", options={"detail": "high"})

    print(f"描述：{result.description}")
    print(f"检测到的对象：{result.objects}")
    print(f"尺寸信息：{result.dimensions}")
    print(f"置信度：{result.confidence}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
