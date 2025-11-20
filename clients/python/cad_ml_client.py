"""
CAD ML Platform Python客户端SDK
提供简单易用的Python接口访问CAD ML Platform服务
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """分析结果数据类"""

    id: str
    timestamp: str
    file_name: str
    file_format: str
    results: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False

    @property
    def part_type(self) -> Optional[str]:
        """获取零件类型"""
        return self.results.get("classification", {}).get("part_type")

    @property
    def confidence(self) -> Optional[float]:
        """获取置信度"""
        return self.results.get("classification", {}).get("confidence")

    @property
    def quality_score(self) -> Optional[float]:
        """获取质量分数"""
        return self.results.get("quality", {}).get("score")

    @property
    def features(self) -> Optional[Dict]:
        """获取特征"""
        return self.results.get("features")


class CADMLClient:
    """CAD ML Platform Python客户端"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        async_mode: bool = False,
    ):
        """
        初始化客户端

        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            async_mode: 是否使用异步模式
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("CADML_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.async_mode = async_mode

        # 同步模式的session
        if not async_mode:
            self.session = requests.Session()
            retry = Retry(
                total=max_retries, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

            if self.api_key:
                self.session.headers["X-API-Key"] = self.api_key

        # 异步模式的session将在运行时创建
        self._async_session = None

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """获取异步session"""
        if self._async_session is None:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._async_session = aiohttp.ClientSession(
                headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._async_session

    def analyze(
        self,
        file: Union[str, Path, BinaryIO],
        extract_features: bool = True,
        classify_parts: bool = True,
        quality_check: bool = True,
        process_recommendation: bool = False,
        calculate_similarity: bool = False,
        reference_id: Optional[str] = None,
    ) -> AnalysisResult:
        """
        分析CAD文件(同步)

        Args:
            file: 文件路径或文件对象
            extract_features: 是否提取特征
            classify_parts: 是否分类零件
            quality_check: 是否质量检查
            process_recommendation: 是否推荐工艺
            calculate_similarity: 是否计算相似度
            reference_id: 参考文件ID(用于相似度计算)

        Returns:
            AnalysisResult: 分析结果
        """
        if self.async_mode:
            raise RuntimeError("Client is in async mode. Use analyze_async() instead.")

        # 准备文件
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            file_name = file_path.name
            with open(file_path, "rb") as f:
                file_data = f.read()
        else:
            file_name = getattr(file, "name", "unknown.dxf")
            file_data = file.read()

        # 准备选项
        options = {
            "extract_features": extract_features,
            "classify_parts": classify_parts,
            "quality_check": quality_check,
            "process_recommendation": process_recommendation,
            "calculate_similarity": calculate_similarity,
            "reference_id": reference_id,
        }

        # 发送请求
        url = f"{self.base_url}/api/v1/analyze"
        files = {"file": (file_name, file_data)}
        data = {"options": json.dumps(options)}

        response = self.session.post(url, files=files, data=data)
        response.raise_for_status()

        result_data = response.json()
        return AnalysisResult(**result_data)

    async def analyze_async(
        self,
        file: Union[str, Path, BinaryIO],
        extract_features: bool = True,
        classify_parts: bool = True,
        quality_check: bool = True,
        process_recommendation: bool = False,
        calculate_similarity: bool = False,
        reference_id: Optional[str] = None,
    ) -> AnalysisResult:
        """
        分析CAD文件(异步)
        """
        # 准备文件
        if isinstance(file, (str, Path)):
            file_path = Path(file)
            file_name = file_path.name
            with open(file_path, "rb") as f:
                file_data = f.read()
        else:
            file_name = getattr(file, "name", "unknown.dxf")
            file_data = file.read()

        # 准备选项
        options = {
            "extract_features": extract_features,
            "classify_parts": classify_parts,
            "quality_check": quality_check,
            "process_recommendation": process_recommendation,
            "calculate_similarity": calculate_similarity,
            "reference_id": reference_id,
        }

        # 发送请求
        url = f"{self.base_url}/api/v1/analyze"
        session = await self._get_async_session()

        data = aiohttp.FormData()
        data.add_field("file", file_data, filename=file_name)
        data.add_field("options", json.dumps(options))

        async with session.post(url, data=data) as response:
            response.raise_for_status()
            result_data = await response.json()

        return AnalysisResult(**result_data)

    def batch_analyze(
        self, files: List[Union[str, Path, BinaryIO]], options: Optional[Dict] = None
    ) -> List[AnalysisResult]:
        """
        批量分析CAD文件

        Args:
            files: 文件列表
            options: 分析选项

        Returns:
            List[AnalysisResult]: 分析结果列表
        """
        results = []
        for file in files:
            try:
                result = self.analyze(file, **(options or {}))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze {file}: {e}")
                results.append(None)

        return results

    def calculate_similarity(
        self, file1: Union[str, Path, BinaryIO], file2: Union[str, Path, BinaryIO]
    ) -> float:
        """
        计算两个CAD文件的相似度

        Args:
            file1: 第一个文件
            file2: 第二个文件

        Returns:
            float: 相似度分数(0-1)
        """
        # 分析两个文件
        result1 = self.analyze(file1, extract_features=True, classify_parts=False)
        result2 = self.analyze(file2, extract_features=True, classify_parts=False)

        # 调用相似度API
        url = f"{self.base_url}/api/v1/similarity/calculate"
        data = {"features1": result1.features, "features2": result2.features}

        response = self.session.post(url, json=data)
        response.raise_for_status()

        return response.json()["similarity"]

    def classify(self, file: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """
        快速分类CAD文件

        Args:
            file: CAD文件

        Returns:
            Dict: 分类结果
        """
        result = self.analyze(
            file,
            extract_features=False,
            classify_parts=True,
            quality_check=False,
            process_recommendation=False,
        )

        return {
            "part_type": result.part_type,
            "confidence": result.confidence,
            "sub_type": result.results.get("classification", {}).get("sub_type"),
            "characteristics": result.results.get("classification", {}).get("characteristics", []),
        }

    def extract_features(self, file: Union[str, Path, BinaryIO]) -> Dict[str, List[float]]:
        """
        提取CAD文件特征

        Args:
            file: CAD文件

        Returns:
            Dict: 特征向量
        """
        result = self.analyze(
            file,
            extract_features=True,
            classify_parts=False,
            quality_check=False,
            process_recommendation=False,
        )

        return result.features

    def check_quality(self, file: Union[str, Path, BinaryIO]) -> Dict[str, Any]:
        """
        检查CAD文件质量

        Args:
            file: CAD文件

        Returns:
            Dict: 质量检查结果
        """
        result = self.analyze(
            file,
            extract_features=False,
            classify_parts=False,
            quality_check=True,
            process_recommendation=False,
        )

        return result.results.get("quality", {})

    def health_check(self) -> bool:
        """
        检查服务健康状态

        Returns:
            bool: 服务是否健康
        """
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()["status"] == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """关闭客户端连接"""
        if not self.async_mode and hasattr(self, "session"):
            self.session.close()

    async def close_async(self):
        """关闭异步客户端连接"""
        if self._async_session:
            await self._async_session.close()
            self._async_session = None

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close_async()


# 便捷函数
def quick_analyze(file_path: str, api_key: Optional[str] = None) -> AnalysisResult:
    """
    快速分析单个文件

    Args:
        file_path: 文件路径
        api_key: API密钥

    Returns:
        AnalysisResult: 分析结果
    """
    with CADMLClient(api_key=api_key) as client:
        return client.analyze(file_path)


def quick_classify(file_path: str, api_key: Optional[str] = None) -> str:
    """
    快速获取零件类型

    Args:
        file_path: 文件路径
        api_key: API密钥

    Returns:
        str: 零件类型
    """
    with CADMLClient(api_key=api_key) as client:
        result = client.classify(file_path)
        return result["part_type"]


# 示例用法
if __name__ == "__main__":
    # 同步示例
    client = CADMLClient(base_url="http://localhost:8000", api_key="your_api_key")

    # 分析单个文件
    result = client.analyze("drawing.dxf")
    print(f"Part type: {result.part_type}")
    print(f"Confidence: {result.confidence}")
    print(f"Quality score: {result.quality_score}")

    # 批量分析
    files = ["file1.dxf", "file2.dxf", "file3.dxf"]
    results = client.batch_analyze(files)
    for r in results:
        if r:
            print(f"{r.file_name}: {r.part_type}")

    # 相似度计算
    similarity = client.calculate_similarity("file1.dxf", "file2.dxf")
    print(f"Similarity: {similarity:.2%}")

    client.close()

    # 异步示例
    async def async_example():
        async_client = CADMLClient(async_mode=True)
        result = await async_client.analyze_async("drawing.dxf")
        print(f"Async result: {result.part_type}")
        await async_client.close_async()

    # asyncio.run(async_example())
