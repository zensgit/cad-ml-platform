"""
Manufacturing Domain Corpus Builder.

Generates contrastive training pairs (anchor/positive/negative) from
manufacturing domain knowledge for embedding fine-tuning.  Covers Chinese
and English terminology across part types, materials, processes, GD&T,
surface finish, tolerances, and common abbreviations.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class ManufacturingCorpusBuilder:
    """Build contrastive training pairs from manufacturing domain knowledge."""

    def __init__(self) -> None:
        self._synonym_pairs: list[tuple[str, str]] = []
        self._hard_negatives: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Synonym / positive-pair builders
    # ------------------------------------------------------------------

    def add_synonym_pairs(self) -> None:
        """Add general manufacturing domain synonym pairs (Chinese)."""
        pairs: list[tuple[str, str]] = [
            # Basic machining terms
            ("法兰盘", "法兰"),
            ("法兰盘", "凸缘盘"),
            ("车削加工", "CNC车削"),
            ("车削加工", "数控车削"),
            ("表面粗糙度Ra3.2", "粗糙度3.2微米"),
            ("IT7公差", "IT7精度等级"),
            ("不锈钢SUS304", "304不锈钢"),
            ("不锈钢SUS304", "0Cr18Ni9不锈钢"),
            ("热处理淬火", "淬火处理"),
            ("热处理淬火", "淬火硬化"),
            ("线切割", "电火花线切割"),
            ("线切割", "WEDM"),
            ("数控铣削", "CNC铣削加工"),
            ("数控铣削", "数控铣床加工"),
            ("电火花加工", "EDM加工"),
            ("电火花加工", "放电加工"),
            ("磨削加工", "研磨加工"),
            ("磨削加工", "grinding"),
            ("钻孔加工", "钻削"),
            ("钻孔加工", "drilling"),
            ("镗孔加工", "镗削"),
            ("镗孔加工", "boring"),
            ("攻丝", "攻螺纹"),
            ("攻丝", "tapping"),
            ("铰孔", "铰削加工"),
            ("拉削", "拉刀加工"),
            ("拉削", "broaching"),
            ("抛光", "精抛加工"),
            ("抛光", "polishing"),
            ("去毛刺", "倒角去毛刺"),
            ("喷砂处理", "喷砂"),
            ("喷砂处理", "sandblasting"),
            ("阳极氧化", "阳极处理"),
            ("阳极氧化", "anodizing"),
            ("电镀", "电镀处理"),
            ("电镀", "electroplating"),
            ("发黑处理", "氧化发黑"),
            ("镀铬", "硬铬电镀"),
            ("镀镍", "化学镀镍"),
            ("渗碳处理", "渗碳淬火"),
            ("渗氮处理", "氮化处理"),
            ("渗氮处理", "nitriding"),
            ("回火处理", "回火"),
            ("回火处理", "tempering"),
            ("正火处理", "正火"),
            ("正火处理", "normalizing"),
            ("退火处理", "退火"),
            ("退火处理", "annealing"),
            ("调质处理", "淬火+回火"),
            # Measurement / inspection
            ("三坐标测量", "CMM测量"),
            ("三坐标测量", "三坐标检测"),
            ("投影仪检测", "光学投影仪"),
            ("粗糙度仪", "表面粗糙度测量仪"),
            ("硬度计", "硬度检测仪"),
            ("卡尺", "游标卡尺"),
            ("千分尺", "外径千分尺"),
            ("塞规", "通止规"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_part_type_pairs(self) -> None:
        """Add part type semantic pairs."""
        pairs: list[tuple[str, str]] = [
            # Flanges
            ("法兰盘", "法兰连接件"),
            ("法兰", "flange"),
            ("焊接法兰", "对焊法兰"),
            ("平焊法兰", "板式法兰"),
            # Shafts
            ("轴", "传动轴"),
            ("轴", "shaft"),
            ("阶梯轴", "多级轴"),
            ("花键轴", "spline shaft"),
            ("空心轴", "中空轴"),
            # Gears
            ("齿轮", "gear"),
            ("齿轮", "传动齿轮"),
            ("直齿轮", "spur gear"),
            ("斜齿轮", "helical gear"),
            ("锥齿轮", "bevel gear"),
            ("蜗轮", "worm gear"),
            # Bearings / supports
            ("轴承座", "bearing housing"),
            ("轴承座", "轴承支架"),
            ("轴承盖", "bearing cap"),
            # Housings
            ("壳体", "外壳"),
            ("壳体", "housing"),
            ("机壳", "机箱壳体"),
            ("箱体", "gear box"),
            ("减速箱体", "减速器壳体"),
            # Brackets
            ("支架", "bracket"),
            ("支架", "安装支架"),
            ("L形支架", "角形支架"),
            # Bushings & sleeves
            ("衬套", "bushing"),
            ("衬套", "轴套"),
            ("导向套", "guide bushing"),
            # Fasteners
            ("螺栓", "bolt"),
            ("螺母", "nut"),
            ("垫圈", "washer"),
            ("销钉", "pin"),
            ("定位销", "dowel pin"),
            ("弹簧", "spring"),
            ("挡圈", "retaining ring"),
            ("键", "key"),
            ("平键", "parallel key"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_material_pairs(self) -> None:
        """Add material synonym pairs."""
        pairs: list[tuple[str, str]] = [
            # Carbon steels
            ("碳钢Q235", "Q235B碳钢"),
            ("碳钢Q235", "Q235普碳钢"),
            ("45号钢", "45#钢"),
            ("45号钢", "C45碳钢"),
            ("45号钢", "S45C中碳钢"),
            ("40Cr钢", "40Cr合金钢"),
            ("40Cr钢", "SCM440"),
            ("20CrMnTi", "渗碳钢20CrMnTi"),
            ("GCr15", "轴承钢GCr15"),
            ("GCr15", "SUJ2轴承钢"),
            ("弹簧钢65Mn", "65Mn弹簧钢"),
            # Stainless steels
            ("不锈钢316L", "316L奥氏体不锈钢"),
            ("不锈钢316L", "SUS316L"),
            ("不锈钢304", "SUS304不锈钢"),
            ("双相不锈钢2205", "S31803双相钢"),
            ("马氏体不锈钢", "420不锈钢"),
            # Aluminum
            ("铝合金6061", "6061-T6铝"),
            ("铝合金6061", "6061铝合金"),
            ("铝合金7075", "7075-T6铝"),
            ("铝合金7075", "超硬铝7075"),
            ("铸铝ADC12", "ADC12压铸铝"),
            ("铝合金5052", "5052防锈铝"),
            ("铝合金2024", "2024硬铝"),
            # Titanium
            ("钛合金TC4", "Ti-6Al-4V"),
            ("钛合金TC4", "TC4钛合金"),
            ("纯钛TA1", "TA1工业纯钛"),
            ("纯钛TA2", "TA2纯钛"),
            # Copper & brass
            ("黄铜H62", "H62黄铜"),
            ("黄铜H62", "HPb59-1铅黄铜"),
            ("紫铜T2", "T2纯铜"),
            ("锡青铜", "QSn6.5-0.1"),
            ("铍铜", "C17200铍铜合金"),
            # Cast iron
            ("灰铸铁HT200", "HT200灰铁"),
            ("灰铸铁HT250", "HT250铸铁"),
            ("球墨铸铁QT500", "QT500-7球铁"),
            ("球墨铸铁", "ductile iron"),
            # Plastics
            ("尼龙PA66", "PA66工程塑料"),
            ("尼龙PA66", "nylon 66"),
            ("聚甲醛POM", "POM赛钢"),
            ("聚甲醛POM", "acetal"),
            ("聚四氟乙烯", "PTFE"),
            ("聚四氟乙烯", "特氟龙"),
            ("聚碳酸酯", "PC塑料"),
            ("PEEK", "聚醚醚酮"),
            ("ABS塑料", "ABS工程塑料"),
            ("聚丙烯", "PP塑料"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_process_pairs(self) -> None:
        """Add manufacturing process synonym pairs."""
        pairs: list[tuple[str, str]] = [
            # CNC processes
            ("五轴加工", "五轴联动加工"),
            ("五轴加工", "5-axis machining"),
            ("三轴铣削", "三轴CNC加工"),
            ("车铣复合", "车铣复合加工中心"),
            ("高速铣削", "HSM加工"),
            ("慢走丝", "慢走丝线切割"),
            ("快走丝", "快走丝线切割"),
            # Casting
            ("压铸", "die casting"),
            ("压铸", "高压铸造"),
            ("砂型铸造", "sand casting"),
            ("精密铸造", "investment casting"),
            ("精密铸造", "熔模铸造"),
            ("失蜡铸造", "蜡模精铸"),
            ("离心铸造", "centrifugal casting"),
            # Forging
            ("锻造", "forging"),
            ("自由锻", "open die forging"),
            ("模锻", "closed die forging"),
            ("冷锻", "cold forging"),
            # Welding
            ("氩弧焊", "TIG焊接"),
            ("氩弧焊", "GTAW焊接"),
            ("二氧化碳焊", "CO2气保焊"),
            ("二氧化碳焊", "MAG焊"),
            ("激光焊接", "laser welding"),
            ("点焊", "电阻点焊"),
            # Sheet metal
            ("钣金加工", "sheet metal"),
            ("钣金折弯", "折弯成型"),
            ("激光切割", "laser cutting"),
            ("冲压加工", "stamping"),
            ("冲压加工", "press forming"),
            ("拉伸成型", "deep drawing"),
            # Additive / other
            ("3D打印", "增材制造"),
            ("3D打印", "additive manufacturing"),
            ("SLM打印", "选择性激光熔化"),
            ("SLA打印", "光固化3D打印"),
            ("注塑成型", "injection molding"),
            ("注塑成型", "注射成型"),
            ("挤出成型", "extrusion"),
            ("粉末冶金", "powder metallurgy"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_gdt_pairs(self) -> None:
        """Add GD&T (Geometric Dimensioning & Tolerancing) terminology pairs."""
        pairs: list[tuple[str, str]] = [
            # Form tolerances
            ("平面度", "flatness"),
            ("平面度", "平面度公差"),
            ("直线度", "straightness"),
            ("直线度", "直线度公差"),
            ("圆度", "roundness"),
            ("圆度", "circularity"),
            ("圆柱度", "cylindricity"),
            ("圆柱度", "圆柱度公差"),
            # Orientation tolerances
            ("平行度", "parallelism"),
            ("平行度", "平行度公差"),
            ("垂直度", "perpendicularity"),
            ("垂直度", "直角度"),
            ("倾斜度", "angularity"),
            ("倾斜度", "倾斜度公差"),
            # Location tolerances
            ("位置度", "position tolerance"),
            ("位置度", "位置度公差"),
            ("同轴度", "concentricity"),
            ("同轴度", "同心度"),
            ("对称度", "symmetry"),
            ("对称度", "对称度公差"),
            # Runout tolerances
            ("圆跳动", "circular runout"),
            ("圆跳动", "径向跳动"),
            ("全跳动", "total runout"),
            ("全跳动", "全跳动公差"),
            # Profile tolerances
            ("线轮廓度", "profile of a line"),
            ("线轮廓度", "线轮廓度公差"),
            ("面轮廓度", "profile of a surface"),
            ("面轮廓度", "面轮廓度公差"),
            # Datum & general GD&T
            ("基准", "datum"),
            ("基准面", "datum plane"),
            ("基准轴", "datum axis"),
            ("最大实体条件", "MMC"),
            ("最大实体条件", "maximum material condition"),
            ("最小实体条件", "LMC"),
            ("最小实体条件", "least material condition"),
            ("自由状态", "free state"),
            ("包容原则", "envelope principle"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_tolerance_surface_pairs(self) -> None:
        """Add tolerance and surface finish terminology pairs."""
        pairs: list[tuple[str, str]] = [
            ("表面粗糙度", "surface roughness"),
            ("Ra0.8", "粗糙度Ra0.8微米"),
            ("Ra1.6", "粗糙度Ra1.6微米"),
            ("Ra3.2", "粗糙度Ra3.2微米"),
            ("Ra6.3", "粗糙度Ra6.3微米"),
            ("Ra12.5", "粗糙度Ra12.5微米"),
            ("光洁度", "表面光洁度"),
            ("Rz", "微观不平度十点高度"),
            ("过盈配合", "interference fit"),
            ("过盈配合", "压入配合"),
            ("间隙配合", "clearance fit"),
            ("间隙配合", "松配合"),
            ("过渡配合", "transition fit"),
            ("H7/g6配合", "H7g6间隙配合"),
            ("H7/k6配合", "H7k6过渡配合"),
            ("H7/p6配合", "H7p6过盈配合"),
            ("尺寸公差", "dimensional tolerance"),
            ("形位公差", "geometric tolerance"),
            ("一般公差", "general tolerance"),
            ("配合公差", "fit tolerance"),
        ]
        self._synonym_pairs.extend(pairs)

    def add_abbreviation_pairs(self) -> None:
        """Add common manufacturing abbreviation pairs."""
        pairs: list[tuple[str, str]] = [
            ("CNC", "计算机数控"),
            ("CAD", "计算机辅助设计"),
            ("CAM", "计算机辅助制造"),
            ("CAE", "计算机辅助工程"),
            ("EDM", "电火花加工"),
            ("WEDM", "线切割"),
            ("CMM", "三坐标测量机"),
            ("DFM", "可制造性设计"),
            ("GD&T", "几何尺寸和公差"),
            ("BOM", "物料清单"),
            ("MRR", "材料去除率"),
            ("SPC", "统计过程控制"),
            ("PPM", "百万分之缺陷率"),
            ("OD", "外径"),
            ("ID", "内径"),
            ("RPM", "转速/每分钟转数"),
        ]
        self._synonym_pairs.extend(pairs)

    # ------------------------------------------------------------------
    # Hard negatives — similar-looking but semantically different
    # ------------------------------------------------------------------

    def add_hard_negatives(self) -> None:
        """Add hard negative pairs (similar surface form, different meaning)."""
        negatives: list[tuple[str, str]] = [
            # Part type confusion
            ("法兰盘", "轴"),
            ("法兰盘", "齿轮"),
            ("轴承", "轴"),
            ("齿轮", "蜗杆"),
            ("螺栓", "销钉"),
            ("弹簧", "垫圈"),
            ("衬套", "壳体"),
            ("支架", "底座"),
            # Material confusion
            ("碳钢Q235", "不锈钢304"),
            ("铝合金6061", "钛合金TC4"),
            ("黄铜H62", "紫铜T2"),
            ("灰铸铁HT200", "球墨铸铁QT500"),
            ("尼龙PA66", "聚甲醛POM"),
            ("碳钢", "合金钢"),
            # Process confusion
            ("车削", "铣削"),
            ("磨削", "抛光"),
            ("淬火", "退火"),
            ("渗碳", "渗氮"),
            ("压铸", "砂型铸造"),
            ("氩弧焊", "激光焊接"),
            ("钣金折弯", "冲压加工"),
            ("3D打印", "注塑成型"),
            # GD&T confusion
            ("平面度", "平行度"),
            ("圆度", "圆柱度"),
            ("同轴度", "对称度"),
            ("圆跳动", "全跳动"),
            ("位置度", "同轴度"),
            # Tolerance/surface confusion
            ("Ra0.8", "Ra6.3"),
            ("过盈配合", "间隙配合"),
            ("IT6", "IT12"),
            ("H7/g6", "H7/p6"),
            # Cross-domain confusion
            ("表面粗糙度", "形位公差"),
            ("热处理", "表面处理"),
            ("锻造", "铸造"),
        ]
        self._hard_negatives.extend(negatives)

    # ------------------------------------------------------------------
    # Build all data
    # ------------------------------------------------------------------

    def build_all(self) -> "ManufacturingCorpusBuilder":
        """Convenience: call every add_* method and return self."""
        self.add_synonym_pairs()
        self.add_part_type_pairs()
        self.add_material_pairs()
        self.add_process_pairs()
        self.add_gdt_pairs()
        self.add_tolerance_surface_pairs()
        self.add_abbreviation_pairs()
        self.add_hard_negatives()
        return self

    def build_training_data(self) -> list[dict[str, Any]]:
        """Return structured training data as anchor/positive/negative triplets.

        Each synonym pair becomes one training sample.  A hard negative is
        randomly assigned to each sample to form a full triplet.  If no hard
        negatives have been added, only anchor/positive pairs are returned.
        """
        training_data: list[dict[str, Any]] = []

        if not self._synonym_pairs:
            return training_data

        for anchor, positive in self._synonym_pairs:
            record: dict[str, Any] = {"anchor": anchor, "positive": positive}
            if self._hard_negatives:
                # Pick a hard negative that is not the anchor or positive
                candidates = [
                    neg
                    for anc, neg in self._hard_negatives
                    if neg != anchor and neg != positive
                ]
                if not candidates:
                    candidates = [neg for _, neg in self._hard_negatives]
                record["negative"] = random.choice(candidates)
            training_data.append(record)

        return training_data

    def export_jsonl(self, output_path: str) -> int:
        """Export training data to JSONL file.

        Returns:
            Number of records written.
        """
        data = self.build_training_data()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            for record in data:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(data)

    # ------------------------------------------------------------------
    # Introspection helpers (useful for tests)
    # ------------------------------------------------------------------

    @property
    def synonym_pairs(self) -> list[tuple[str, str]]:
        return list(self._synonym_pairs)

    @property
    def hard_negatives(self) -> list[tuple[str, str]]:
        return list(self._hard_negatives)

    def domain_coverage(self) -> dict[str, bool]:
        """Check whether the corpus covers major domain categories.

        Returns a dict mapping category name to a boolean indicating
        whether at least one pair touches that category.
        """
        all_texts = set()
        for a, b in self._synonym_pairs:
            all_texts.add(a)
            all_texts.add(b)

        joined = " ".join(all_texts)

        return {
            "materials": any(
                kw in joined
                for kw in ["钢", "铝", "钛", "铜", "铸铁", "尼龙", "steel", "aluminum"]
            ),
            "processes": any(
                kw in joined
                for kw in ["加工", "铣削", "车削", "铸造", "焊接", "CNC", "EDM"]
            ),
            "gdt": any(
                kw in joined
                for kw in ["平面度", "圆度", "位置度", "flatness", "runout"]
            ),
            "parts": any(
                kw in joined
                for kw in ["法兰", "轴", "齿轮", "轴承", "flange", "shaft", "gear"]
            ),
            "surface_finish": any(
                kw in joined for kw in ["粗糙度", "Ra", "roughness", "光洁度"]
            ),
            "tolerances": any(
                kw in joined for kw in ["公差", "配合", "IT", "tolerance", "fit"]
            ),
        }
