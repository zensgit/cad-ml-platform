#!/usr/bin/env python3
"""
渲染可疑样本的DXF文件，方便人工审核
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REAL_DATA_DIR = Path("data/training_v7")
OUTPUT_DIR = Path("claudedocs/suspicious_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可疑样本及其预测
SUSPICIOUS_SAMPLES = [
    {"file": "其他/old_0033.dxf", "label": "其他", "pred": "轴类"},
    {"file": "其他/old_0085.dxf", "label": "其他", "pred": "传动件"},
    {"file": "连接件/old_0008.dxf", "label": "连接件", "pred": "传动件"},
    {"file": "其他/new_0208.dxf", "label": "其他", "pred": "壳体类", "note": "与old_0086.dxf重复"},
]


def render_dxf(dxf_path: str, output_path: str, title: str):
    """渲染DXF文件为PNG图片"""
    try:
        import ezdxf
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig = plt.figure(figsize=(10, 10), dpi=150)
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')

        all_x, all_y = [], []

        for entity in msp:
            try:
                etype = entity.dxftype()
                if etype == "LINE":
                    x1, y1 = entity.dxf.start.x, entity.dxf.start.y
                    x2, y2 = entity.dxf.end.x, entity.dxf.end.y
                    ax.plot([x1, x2], [y1, y2], 'b-', linewidth=0.5)
                    all_x.extend([x1, x2])
                    all_y.extend([y1, y2])
                elif etype == "CIRCLE":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    circle = plt.Circle((cx, cy), r, fill=False, color='blue', linewidth=0.5)
                    ax.add_patch(circle)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype == "ARC":
                    cx, cy = entity.dxf.center.x, entity.dxf.center.y
                    r = entity.dxf.radius
                    from matplotlib.patches import Arc
                    arc = Arc((cx, cy), 2*r, 2*r, angle=0,
                              theta1=entity.dxf.start_angle, theta2=entity.dxf.end_angle,
                              color='blue', linewidth=0.5)
                    ax.add_patch(arc)
                    all_x.extend([cx-r, cx+r])
                    all_y.extend([cy-r, cy+r])
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, 'get_points'):
                        pts = list(entity.get_points())
                        if len(pts) >= 2:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            ax.plot(xs, ys, 'b-', linewidth=0.5)
                            all_x.extend(xs)
                            all_y.extend(ys)
            except:
                pass

        if all_x and all_y:
            margin = 0.05
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

        ax.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"✓ 保存: {output_path}")
        return True
    except Exception as e:
        print(f"✗ 错误 {dxf_path}: {e}")
        return False


def main():
    print("=" * 60)
    print("渲染可疑样本")
    print("=" * 60)

    for i, sample in enumerate(SUSPICIOUS_SAMPLES, 1):
        file_path = REAL_DATA_DIR / sample["file"]
        title = f"{sample['file']}\n标注: {sample['label']} → 模型预测: {sample['pred']}"
        if "note" in sample:
            title += f"\n({sample['note']})"

        output_name = f"sample_{i}_{Path(sample['file']).stem}.png"
        output_path = OUTPUT_DIR / output_name

        print(f"\n{i}. {sample['file']}")
        print(f"   标注: {sample['label']} → 预测: {sample['pred']}")
        render_dxf(str(file_path), str(output_path), title)

    print(f"\n图片已保存到: {OUTPUT_DIR}/")
    print("\n各类别特征参考:")
    print("  轴类 - 细长形状，圆形截面，同心圆多")
    print("  传动件 - 齿轮、链轮、皮带轮等，有齿形或轮廓")
    print("  壳体类 - 复杂外形，有腔体、孔位")
    print("  连接件 - 螺栓、螺母、法兰等，相对简单")
    print("  其他 - 不属于以上类别")


if __name__ == "__main__":
    main()
