# 机械装配理解 AI 方案（Assembly Understanding Plan）

## 1. 目标与范围
- 目标：让系统“理解”装配关系与工作原理，而非仅做识别/标注。
- 输出：装配图（Assembly Graph）、功能解释、传动与约束关系、URDF/SDF 导出、仿真验证报告。

## 2. 表示与数据结构（Assembly Graph）
- 节点（Part/Subsystem）：id、类别、材料、关键尺寸、功能标签。
- 边（Relations）：约束/接触/传动（Revolute/Prismatic/Fixed/Gear/Belt/Chain），包含轴线/间隙/模数/传动比/自由度。
- 附加：BOM、工序、容差链、润滑/密封、维护要点。

## 3. 数据与标注
- CAD 导出（STEP/SAT + mates）→ 自动生成装配图标签（关节类型、配合轴/面、自由度）。
- 2D 图纸（DXF）：尺寸、公差、符号抽取；技术要求文本抽取工艺/功能。
- 文档增强：术语表+常见机构知识库（齿轮/轴承/键/花键/皮带/链传动模板）。

## 4. 感知层（Perception）
- 几何解析：OpenCascade/FreeCAD 提取 B-Rep、孔/键槽/齿形/配合面、主轴线与基准。
- 视觉/OCR：图像/截图的尺寸、公差、符号检测与文本抽取。
- 表示学习：点云/网格嵌入（PointNet++/DGCNN）+ 图神经网络（PyG/DGL）做边分类（关节/传动）与子装识别。

## 5. 机理与推理（Mechanics-aware Reasoning）
- 规则库：齿轮啮合、皮带/链传动、轴承支撑、配合/间隙、键/花键约束等模板与约束检查。
- 运动学：由装配图自动生成 URDF/SDF，识别自由度、闭环与过约束。
- 因果/功能推理：沿“动力/运动”路径推断功能、传动比、能量流；异常检测（打滑/干涉/过约束）。
- LLM 协同：LLM 负责解释与对话，但结论需引用装配图证据（节点/边/参数）。

## 6. 仿真与验证
- 仿真引擎：PyBullet/MuJoCo/Chrono；从 URDF/SDF 生成关节与惯量近似，运行运动学/简化动力学。
- 回环校验：仿真与规则矛盾时输出不一致点（如中心距/模数不匹配、轴承方向错误）。

## 7. 系统架构与集成
- 分层：解析/特征 → 图构建 → 学习/规则 → 仿真 → 解释/可视化。
- 存储：装配图（图数据库/JSON 文档）+ 向量索引（相似装配检索）+ 机构知识库。
- API 扩展（与现有 `/api/v1` 对齐）：
  - `POST /api/v1/assembly/analyze`（上传 CAD/图像 → 装配图 + 解释）
  - `GET /api/v1/assembly/{id}/graph`（装配图 JSON）
  - `GET /api/v1/assembly/{id}/urdf`（导出 URDF）
  - `POST /api/v1/assembly/{id}/simulate`（仿真验证与报告）

## 8. 实施路线（里程碑）
- P1（4–6周）MVP：
  - 从 STEP+mates 提取装配图；实现齿轮/轴承/键连接规则；URDF 导出；基础仿真；RAG 解释。
  - 验收：3 套典型传动装置正确生成装配图并通过仿真；接口与示例文档完善。
- P2（6–8周）：
  - 图学习（边/子装分类、功能标签预测）；2D/3D 融合抽取；错误检测与修复建议。
  - 验收：公开/自建数据集上 >85% 关系识别准确率；复杂装配通过闭环检查。
- P3（>8周）：
  - 公差链与失效模式（FMEA）提示；成本/工艺联动评估；交互式解释与可视化。

## 9. 技术选型与数据源
- 几何/CAD：OpenCascade/FreeCAD、ezdxf；URDF/SDF 生成脚本。
- 学习：PyTorch Geometric/DGL、Faiss（相似装配检索）。
- 仿真：PyBullet/MuJoCo/Project Chrono。
- 数据：Fusion 360 Gallery（草图/约束）、PartNet/PartNet-Mobility（关节）、ABC/ShapeNet（几何），结合自建 mates 自动标注集。

## 10. 与现有平台对齐
- 与 `src/main.py` 的健康/就绪/指标一致；新增路由置于 `src/api/v1/assembly.py` 并在 `src/api/__init__.py` 注册。
- 结果对象采用与 `VisionAnalysisResult` 相同风格的 Pydantic 模型，保证一致的响应结构与客户端体验。
