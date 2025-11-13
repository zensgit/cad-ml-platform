"""
视觉识别API端点
支持上传图片/截图/照片来识别CAD图纸内容
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel, Field
import json

from src.core.vision_analyzer import VisionAnalyzer, VisionProvider
from src.core.cad_understanding import CADUnderstanding
from src.api.dependencies import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class VisionAnalysisResult(BaseModel):
    """视觉分析结果"""

    # 基础识别结果
    what_is_it: str = Field(description="这是什么零件/产品")
    category: str = Field(description="分类：机械零件/电子元件/建筑图纸等")
    part_type: str = Field(description="具体类型：轴/齿轮/轴承/板材等")

    # 详细理解
    purpose: str = Field(description="用途和功能")
    materials: list[str] = Field(description="可能的材料")
    manufacturing: Dict[str, Any] = Field(description="制造工艺建议")

    # 技术参数
    dimensions: Dict[str, Any] = Field(description="识别到的尺寸")
    features: list[str] = Field(description="关键特征")
    specifications: Dict[str, Any] = Field(description="技术规格")

    # 智能分析
    similar_products: list[str] = Field(description="类似产品")
    industry_application: str = Field(description="行业应用")
    complexity_level: str = Field(description="复杂度：简单/中等/复杂")
    estimated_cost: Optional[str] = Field(description="成本估算")

    # 元数据
    confidence: float = Field(description="识别置信度")
    analysis_method: str = Field(description="分析方法")


class CADUnderstanding:
    """
    CAD图纸理解引擎
    将视觉识别结果转化为对设计内容的理解
    """

    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.pattern_matcher = self._init_pattern_matcher()

    def _load_knowledge_base(self) -> Dict:
        """加载零件知识库"""
        return {
            "shaft_types": {
                "stepped_shaft": {
                    "features": ["多个直径段", "台阶", "轴肩"],
                    "applications": ["传动系统", "电机轴", "减速器"],
                    "materials": ["45号钢", "40Cr", "不锈钢304"],
                    "manufacturing": ["车削", "磨削", "热处理"]
                },
                "splined_shaft": {
                    "features": ["花键", "键槽", "渐开线"],
                    "applications": ["动力传输", "精确定位", "高扭矩传递"],
                    "materials": ["40Cr", "20CrMnTi", "42CrMo"],
                    "manufacturing": ["滚齿", "插齿", "磨齿"]
                }
            },
            "gear_types": {
                "spur_gear": {
                    "features": ["直齿", "模数", "齿数标注"],
                    "applications": ["减速器", "变速箱", "传动装置"],
                    "materials": ["20CrMnTi", "40Cr", "尼龙"],
                    "manufacturing": ["滚齿", "插齿", "热处理", "磨齿"]
                },
                "helical_gear": {
                    "features": ["斜齿", "螺旋角", "轴向力"],
                    "applications": ["高速传动", "平稳传动", "大功率传输"],
                    "materials": ["20CrMnTi", "17CrNiMo6"],
                    "manufacturing": ["滚齿", "磨齿", "渗碳淬火"]
                }
            },
            "plate_types": {
                "base_plate": {
                    "features": ["平板", "安装孔", "定位孔"],
                    "applications": ["设备底座", "安装平台", "工作台"],
                    "materials": ["Q235", "铝合金", "铸铁"],
                    "manufacturing": ["铣削", "钻孔", "攻丝"]
                },
                "bracket": {
                    "features": ["L型", "加强筋", "连接孔"],
                    "applications": ["支撑", "固定", "连接"],
                    "materials": ["Q235", "不锈钢", "铝合金"],
                    "manufacturing": ["折弯", "焊接", "冲压"]
                }
            },
            "housing_types": {
                "bearing_housing": {
                    "features": ["轴承座孔", "油封槽", "散热筋"],
                    "applications": ["轴承支撑", "传动系统", "泵类设备"],
                    "materials": ["HT200", "QT450", "铝合金"],
                    "manufacturing": ["铸造", "加工中心", "镗削"]
                },
                "gearbox_housing": {
                    "features": ["多个轴承孔", "油路", "密封面"],
                    "applications": ["减速器", "变速箱", "传动装置"],
                    "materials": ["HT250", "QT500", "铝合金ADC12"],
                    "manufacturing": ["铸造", "加工中心", "精镗"]
                }
            }
        }

    def _init_pattern_matcher(self):
        """初始化模式匹配器"""
        return {
            "keywords": {
                "shaft": ["轴", "shaft", "spindle", "arbor"],
                "gear": ["齿轮", "gear", "pinion", "齿"],
                "bearing": ["轴承", "bearing", "滚动", "滑动"],
                "plate": ["板", "plate", "基板", "底板"],
                "housing": ["箱体", "housing", "壳体", "外壳"]
            },
            "visual_patterns": {
                "cylindrical": "可能是轴类零件",
                "circular_with_teeth": "可能是齿轮",
                "rectangular_with_holes": "可能是板类零件",
                "complex_cavity": "可能是箱体类零件"
            }
        }

    async def understand_cad_image(
        self,
        vision_result: Dict[str, Any]
    ) -> VisionAnalysisResult:
        """
        理解CAD图纸内容

        Args:
            vision_result: 视觉识别原始结果

        Returns:
            VisionAnalysisResult: 结构化的理解结果
        """

        # 1. 初步分类
        category, part_type = self._classify_drawing(vision_result)

        # 2. 提取关键特征
        features = self._extract_key_features(vision_result)

        # 3. 匹配知识库
        knowledge = self._match_knowledge(category, part_type, features)

        # 4. 生成理解结果
        understanding = VisionAnalysisResult(
            what_is_it=self._generate_description(category, part_type, features),
            category=category,
            part_type=part_type,
            purpose=knowledge.get("applications", ["通用机械零件"])[0],
            materials=knowledge.get("materials", ["钢材"]),
            manufacturing=self._suggest_manufacturing(part_type, features),
            dimensions=self._extract_dimensions(vision_result),
            features=features,
            specifications=self._extract_specifications(vision_result),
            similar_products=self._find_similar_products(part_type),
            industry_application=self._determine_industry(category, part_type),
            complexity_level=self._assess_complexity(features),
            estimated_cost=self._estimate_cost(part_type, features),
            confidence=vision_result.get("confidence", 0.85),
            analysis_method="Vision + Knowledge Base + Pattern Matching"
        )

        return understanding

    def _classify_drawing(self, vision_result: Dict) -> tuple[str, str]:
        """分类图纸"""
        text = vision_result.get("description", "").lower()
        text += " " + vision_result.get("text", "").lower()

        # 检测关键词
        if any(kw in text for kw in ["轴", "shaft", "spindle"]):
            if "花键" in text or "spline" in text:
                return "机械零件", "花键轴"
            elif "阶梯" in text or "stepped" in text:
                return "机械零件", "阶梯轴"
            else:
                return "机械零件", "传动轴"

        elif any(kw in text for kw in ["齿轮", "gear", "齿"]):
            if "斜齿" in text or "helical" in text:
                return "机械零件", "斜齿轮"
            else:
                return "机械零件", "直齿轮"

        elif any(kw in text for kw in ["板", "plate", "基板"]):
            return "机械零件", "板材"

        elif any(kw in text for kw in ["箱体", "housing", "壳体"]):
            return "机械零件", "箱体"

        else:
            return "机械零件", "通用零件"

    def _extract_key_features(self, vision_result: Dict) -> list[str]:
        """提取关键特征"""
        features = []

        text = vision_result.get("description", "")

        # 特征关键词
        feature_keywords = {
            "孔": "安装孔",
            "键槽": "键槽",
            "螺纹": "螺纹",
            "倒角": "倒角",
            "圆角": "圆角",
            "花键": "花键",
            "齿": "齿形",
            "凹槽": "凹槽",
            "台阶": "台阶",
            "轴肩": "轴肩"
        }

        for keyword, feature_name in feature_keywords.items():
            if keyword in text:
                features.append(feature_name)

        # 从识别的对象中提取
        objects = vision_result.get("objects", [])
        for obj in objects:
            if obj.get("class") == "hole":
                features.append("孔特征")
            elif obj.get("class") == "thread":
                features.append("螺纹特征")

        return features

    def _match_knowledge(
        self,
        category: str,
        part_type: str,
        features: list[str]
    ) -> Dict:
        """匹配知识库"""

        # 简化匹配逻辑
        if "轴" in part_type:
            if "花键" in features:
                return self.knowledge_base["shaft_types"]["splined_shaft"]
            else:
                return self.knowledge_base["shaft_types"]["stepped_shaft"]

        elif "齿轮" in part_type:
            if "斜齿" in part_type:
                return self.knowledge_base["gear_types"]["helical_gear"]
            else:
                return self.knowledge_base["gear_types"]["spur_gear"]

        elif "板" in part_type:
            return self.knowledge_base["plate_types"]["base_plate"]

        elif "箱体" in part_type:
            return self.knowledge_base["housing_types"]["gearbox_housing"]

        return {}

    def _generate_description(
        self,
        category: str,
        part_type: str,
        features: list[str]
    ) -> str:
        """生成零件描述"""

        feature_str = "、".join(features[:3]) if features else "标准特征"

        descriptions = {
            "传动轴": f"这是一个{part_type}，具有{feature_str}，用于动力传输",
            "花键轴": f"这是一个{part_type}，具有{feature_str}，用于精确传动定位",
            "直齿轮": f"这是一个{part_type}，具有{feature_str}，用于减速传动",
            "斜齿轮": f"这是一个{part_type}，具有{feature_str}，用于高速平稳传动",
            "板材": f"这是一个{part_type}零件，具有{feature_str}，用于安装或支撑",
            "箱体": f"这是一个{part_type}零件，具有{feature_str}，用于容纳其他部件"
        }

        return descriptions.get(part_type, f"这是一个{category}的{part_type}")

    def _suggest_manufacturing(
        self,
        part_type: str,
        features: list[str]
    ) -> Dict:
        """建议制造工艺"""

        processes = {
            "主要工艺": [],
            "辅助工艺": [],
            "后处理": []
        }

        # 根据零件类型推荐工艺
        if "轴" in part_type:
            processes["主要工艺"] = ["车削", "铣削"]
            processes["辅助工艺"] = ["钻孔", "攻丝"]
            processes["后处理"] = ["热处理", "表面处理"]

        elif "齿轮" in part_type:
            processes["主要工艺"] = ["滚齿", "插齿"]
            processes["辅助工艺"] = ["车削", "钻孔"]
            processes["后处理"] = ["热处理", "磨齿"]

        elif "板" in part_type:
            processes["主要工艺"] = ["激光切割", "冲压"]
            processes["辅助工艺"] = ["折弯", "钻孔"]
            processes["后处理"] = ["去毛刺", "表面处理"]

        elif "箱体" in part_type:
            processes["主要工艺"] = ["铸造", "加工中心"]
            processes["辅助工艺"] = ["镗削", "钻孔"]
            processes["后处理"] = ["打磨", "喷漆"]

        return processes

    def _extract_dimensions(self, vision_result: Dict) -> Dict:
        """提取尺寸信息"""
        # 从OCR文本中提取尺寸
        import re

        dimensions = {}
        text = vision_result.get("text", "")

        # 查找尺寸模式 (例如: Φ20, R10, 100±0.1)
        diameter_pattern = r'[Φφ](\d+(?:\.\d+)?)'
        radius_pattern = r'R(\d+(?:\.\d+)?)'
        length_pattern = r'(\d+(?:\.\d+)?)\s*(?:mm|MM|毫米)'

        diameters = re.findall(diameter_pattern, text)
        radii = re.findall(radius_pattern, text)
        lengths = re.findall(length_pattern, text)

        if diameters:
            dimensions["直径"] = [f"Φ{d}" for d in diameters]
        if radii:
            dimensions["半径"] = [f"R{r}" for r in radii]
        if lengths:
            dimensions["长度"] = [f"{l}mm" for l in lengths]

        return dimensions

    def _extract_specifications(self, vision_result: Dict) -> Dict:
        """提取技术规格"""
        specs = {
            "公差等级": "IT7",  # 示例
            "表面粗糙度": "Ra1.6",
            "硬度要求": "HRC45-50",
            "热处理": "调质处理"
        }

        # TODO: 从实际识别结果中提取

        return specs

    def _find_similar_products(self, part_type: str) -> list[str]:
        """查找类似产品"""
        similar = {
            "传动轴": ["电机轴", "主轴", "心轴"],
            "花键轴": ["渐开线花键轴", "矩形花键轴", "三角花键轴"],
            "直齿轮": ["标准齿轮", "变位齿轮", "内齿轮"],
            "斜齿轮": ["人字齿轮", "螺旋齿轮", "锥齿轮"],
            "板材": ["底板", "支撑板", "连接板"],
            "箱体": ["减速器箱体", "轴承座", "泵体"]
        }

        return similar.get(part_type, ["通用机械零件"])

    def _determine_industry(self, category: str, part_type: str) -> str:
        """确定行业应用"""
        industries = {
            "传动轴": "机械传动、汽车工业、工程机械",
            "齿轮": "减速器、变速箱、精密仪器",
            "板材": "设备制造、钢结构、机械加工",
            "箱体": "机械设备、汽车零部件、泵阀行业"
        }

        for key, value in industries.items():
            if key in part_type:
                return value

        return "通用机械制造"

    def _assess_complexity(self, features: list[str]) -> str:
        """评估复杂度"""
        feature_count = len(features)

        if feature_count <= 3:
            return "简单"
        elif feature_count <= 6:
            return "中等"
        else:
            return "复杂"

    def _estimate_cost(self, part_type: str, features: list[str]) -> str:
        """估算成本"""
        # 基础成本估算逻辑
        base_costs = {
            "传动轴": 200,
            "花键轴": 500,
            "直齿轮": 300,
            "斜齿轮": 450,
            "板材": 100,
            "箱体": 800
        }

        base = base_costs.get(part_type, 200)

        # 特征复杂度加成
        feature_cost = len(features) * 50

        total = base + feature_cost

        return f"约 {total}-{total*1.5} 元（参考价）"


@router.post("/analyze-image", response_model=VisionAnalysisResult)
async def analyze_cad_image(
    image: UploadFile = File(..., description="CAD图纸图片/截图/照片"),
    provider: str = "openai",
    api_key: str = Depends(get_api_key)
):
    """
    通过图片识别CAD图纸内容

    支持：
    - CAD软件截图
    - 手机拍摄的图纸照片
    - 扫描的图纸图片
    - PNG/JPG/JPEG格式

    返回：
    - 零件类型识别
    - 功能用途分析
    - 制造工艺建议
    - 材料推荐
    - 成本估算
    """

    try:
        # 读取图片
        image_data = await image.read()

        # 初始化视觉分析器
        vision_provider = VisionProvider[provider.upper()]
        analyzer = VisionAnalyzer(provider=vision_provider)

        # 执行视觉分析
        vision_result = await analyzer.analyze_image(
            image_data,
            task="cad",
            options={"detail": "high"}
        )

        # 转换为字典
        vision_dict = {
            "description": vision_result.description,
            "objects": vision_result.objects,
            "text": vision_result.text,
            "drawings": vision_result.drawings,
            "dimensions": vision_result.dimensions,
            "confidence": vision_result.confidence
        }

        # 理解CAD内容
        understanding_engine = CADUnderstanding()
        result = await understanding_engine.understand_cad_image(vision_dict)

        logger.info(f"Successfully analyzed image: {image.filename}")

        return result

    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/compare-images")
async def compare_cad_images(
    image1: UploadFile = File(..., description="第一张CAD图纸"),
    image2: UploadFile = File(..., description="第二张CAD图纸"),
    api_key: str = Depends(get_api_key)
):
    """
    比较两张CAD图纸的相似度

    返回：
    - 相似度分数
    - 相同点
    - 差异点
    - 可能是同一零件的不同版本
    """

    try:
        # 分析两张图片
        understanding_engine = CADUnderstanding()
        analyzer = VisionAnalyzer(provider=VisionProvider.OPENAI)

        # 分析第一张
        image1_data = await image1.read()
        result1 = await analyzer.analyze_image(image1_data, task="cad")
        understanding1 = await understanding_engine.understand_cad_image({
            "description": result1.description,
            "text": result1.text,
            "confidence": result1.confidence
        })

        # 分析第二张
        image2_data = await image2.read()
        result2 = await analyzer.analyze_image(image2_data, task="cad")
        understanding2 = await understanding_engine.understand_cad_image({
            "description": result2.description,
            "text": result2.text,
            "confidence": result2.confidence
        })

        # 比较
        similarity = _calculate_similarity(understanding1, understanding2)

        return {
            "similarity_score": similarity,
            "image1_analysis": understanding1,
            "image2_analysis": understanding2,
            "comparison": {
                "same_category": understanding1.category == understanding2.category,
                "same_part_type": understanding1.part_type == understanding2.part_type,
                "feature_overlap": len(set(understanding1.features) & set(understanding2.features)),
                "likely_same_part": similarity > 0.8
            }
        }

    except Exception as e:
        logger.error(f"Image comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


def _calculate_similarity(
    understanding1: VisionAnalysisResult,
    understanding2: VisionAnalysisResult
) -> float:
    """计算两个理解结果的相似度"""

    score = 0.0

    # 类别相同 +0.3
    if understanding1.category == understanding2.category:
        score += 0.3

    # 类型相同 +0.3
    if understanding1.part_type == understanding2.part_type:
        score += 0.3

    # 特征重叠度 +0.2
    if understanding1.features and understanding2.features:
        overlap = len(set(understanding1.features) & set(understanding2.features))
        total = len(set(understanding1.features) | set(understanding2.features))
        if total > 0:
            score += 0.2 * (overlap / total)

    # 材料相同 +0.1
    if set(understanding1.materials) & set(understanding2.materials):
        score += 0.1

    # 复杂度相同 +0.1
    if understanding1.complexity_level == understanding2.complexity_level:
        score += 0.1

    return min(score, 1.0)


# 使用示例
"""
使用方式：

1. 上传CAD截图识别：
curl -X POST "http://localhost:8000/api/v1/vision/analyze-image" \
  -H "X-API-Key: your_api_key" \
  -F "image=@cad_screenshot.png" \
  -F "provider=openai"

2. 返回结果示例：
{
  "what_is_it": "这是一个阶梯轴，具有键槽、轴肩、倒角，用于动力传输",
  "category": "机械零件",
  "part_type": "阶梯轴",
  "purpose": "传动系统",
  "materials": ["45号钢", "40Cr", "不锈钢304"],
  "manufacturing": {
    "主要工艺": ["车削", "铣削"],
    "辅助工艺": ["钻孔", "攻丝"],
    "后处理": ["热处理", "表面处理"]
  },
  "dimensions": {
    "直径": ["Φ20", "Φ25", "Φ30"],
    "长度": ["150mm"]
  },
  "features": ["键槽", "轴肩", "倒角", "退刀槽"],
  "complexity_level": "中等",
  "estimated_cost": "约 400-600 元（参考价）",
  "confidence": 0.92
}
"""