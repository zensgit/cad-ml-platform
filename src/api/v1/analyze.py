"""
CAD文件分析API端点
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from pydantic import BaseModel, Field
import json

from src.core.analyzer import CADAnalyzer
from src.core.feature_extractor import FeatureExtractor
from src.adapters.factory import AdapterFactory
from src.utils.cache import cache_result, get_cached_result
from src.api.dependencies import get_api_key

logger = logging.getLogger(__name__)

router = APIRouter()


class AnalysisOptions(BaseModel):
    """分析选项"""
    extract_features: bool = Field(default=True, description="是否提取特征")
    classify_parts: bool = Field(default=True, description="是否分类零件")
    calculate_similarity: bool = Field(default=False, description="是否计算相似度")
    reference_id: Optional[str] = Field(default=None, description="参考文件ID")
    quality_check: bool = Field(default=True, description="是否质量检查")
    process_recommendation: bool = Field(default=False, description="是否推荐工艺")


class AnalysisResult(BaseModel):
    """分析结果"""
    id: str = Field(description="分析ID")
    timestamp: datetime = Field(description="分析时间")
    file_name: str = Field(description="文件名")
    file_format: str = Field(description="文件格式")
    results: Dict[str, Any] = Field(description="分析结果")
    processing_time: float = Field(description="处理时间(秒)")
    cache_hit: bool = Field(default=False, description="是否缓存命中")


@router.post("/", response_model=AnalysisResult)
async def analyze_cad_file(
    file: UploadFile = File(..., description="CAD文件"),
    options: str = Form(default='{"extract_features": true, "classify_parts": true}'),
    api_key: str = Depends(get_api_key)
):
    """
    分析CAD文件

    支持格式:
    - DXF (AutoCAD)
    - DWG (通过转换)
    - STEP
    - IGES
    - STL
    """
    start_time = datetime.utcnow()
    analysis_id = str(uuid.uuid4())

    try:
        # 解析选项
        analysis_options = AnalysisOptions(**json.loads(options))

        # 检查缓存
        cache_key = f"analysis:{file.filename}:{options}"
        cached = await get_cached_result(cache_key)
        if cached:
            logger.info(f"Cache hit for {file.filename}")
            return AnalysisResult(
                id=analysis_id,
                timestamp=start_time,
                file_name=file.filename,
                file_format=file.filename.split('.')[-1].upper(),
                results=cached,
                processing_time=0.1,
                cache_hit=True
            )

        # 读取文件内容
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        # 获取文件格式
        file_format = file.filename.split('.')[-1].lower()
        if file_format not in ['dxf', 'dwg', 'step', 'stp', 'iges', 'igs', 'stl']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_format}"
            )

        # 使用适配器转换格式
        adapter = AdapterFactory.get_adapter(file_format)
        unified_data = await adapter.convert(content)

        # 创建分析器
        analyzer = CADAnalyzer()
        results = {}

        # 执行分析
        if analysis_options.extract_features:
            extractor = FeatureExtractor()
            features = await extractor.extract(unified_data)
            results['features'] = {
                'geometric': features['geometric'].tolist(),
                'semantic': features['semantic'].tolist(),
                'dimension': len(features['geometric']) + len(features['semantic'])
            }

        if analysis_options.classify_parts:
            classification = await analyzer.classify_part(unified_data)
            results['classification'] = {
                'part_type': classification['type'],
                'confidence': classification['confidence'],
                'sub_type': classification.get('sub_type'),
                'characteristics': classification.get('characteristics', [])
            }

        if analysis_options.quality_check:
            quality = await analyzer.check_quality(unified_data)
            results['quality'] = {
                'score': quality['score'],
                'issues': quality.get('issues', []),
                'suggestions': quality.get('suggestions', [])
            }

        if analysis_options.process_recommendation:
            process = await analyzer.recommend_process(unified_data)
            results['process'] = {
                'recommended_process': process['primary'],
                'alternatives': process.get('alternatives', []),
                'parameters': process.get('parameters', {})
            }

        if analysis_options.calculate_similarity and analysis_options.reference_id:
            # TODO: 实现相似度计算
            results['similarity'] = {
                'reference_id': analysis_options.reference_id,
                'score': 0.0,
                'details': {}
            }

        # 添加统计信息
        results['statistics'] = {
            'entity_count': unified_data.get('entity_count', 0),
            'layer_count': unified_data.get('layer_count', 0),
            'bounding_box': unified_data.get('bounding_box', {}),
            'complexity': unified_data.get('complexity', 'medium')
        }

        # 缓存结果
        await cache_result(cache_key, results, ttl=3600)

        # 计算处理时间
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Analysis completed for {file.filename} in {processing_time:.2f}s")

        return AnalysisResult(
            id=analysis_id,
            timestamp=start_time,
            file_name=file.filename,
            file_format=file_format.upper(),
            results=results,
            processing_time=processing_time,
            cache_hit=False
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options format")
    except Exception as e:
        logger.error(f"Analysis failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/batch")
async def batch_analyze(
    files: list[UploadFile] = File(..., description="多个CAD文件"),
    options: str = Form(default='{"extract_features": true}'),
    api_key: str = Depends(get_api_key)
):
    """批量分析CAD文件"""
    results = []

    for file in files:
        try:
            result = await analyze_cad_file(file, options, api_key)
            results.append(result)
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e)
            })

    return {
        "total": len(files),
        "successful": len([r for r in results if 'error' not in r]),
        "failed": len([r for r in results if 'error' in r]),
        "results": results
    }


@router.get("/{analysis_id}")
async def get_analysis_result(
    analysis_id: str,
    api_key: str = Depends(get_api_key)
):
    """获取分析结果"""
    # TODO: 从数据库或缓存获取历史分析结果
    cache_key = f"analysis_result:{analysis_id}"
    result = await get_cached_result(cache_key)

    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return result