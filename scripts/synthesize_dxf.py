#!/usr/bin/env python3
"""
合成DXF训练数据

为每个类别生成具有代表性特征的DXF文件
"""

import random
import math
from pathlib import Path
import ezdxf
from ezdxf import units

OUTPUT_DIR = Path("data/synthetic_dxf")

def create_shaft(doc, msp, params):
    """生成轴类图纸 - 同心圆、阶梯轴、键槽"""
    cx, cy = 0, 0
    
    # 主轴轮廓 (多段阶梯)
    stages = random.randint(2, 5)
    x = -params['length'] / 2
    
    for i in range(stages):
        stage_len = params['length'] / stages
        r = params['radius'] * (0.5 + 0.5 * random.random())
        
        # 上下轮廓线
        msp.add_line((x, r), (x + stage_len, r))
        msp.add_line((x, -r), (x + stage_len, -r))
        
        # 阶梯连接
        if i > 0:
            msp.add_line((x, prev_r), (x, r))
            msp.add_line((x, -prev_r), (x, -r))
        
        prev_r = r
        x += stage_len
    
    # 端面封闭
    msp.add_line((-params['length']/2, prev_r), (-params['length']/2, -prev_r))
    msp.add_line((params['length']/2, r), (params['length']/2, -r))
    
    # 中心线
    msp.add_line((-params['length']/2 - 10, 0), (params['length']/2 + 10, 0), 
                 dxfattribs={'layer': 'CENTER'})
    
    # 同心圆 (端面视图偏移显示)
    offset_x = params['length'] + 50
    for i in range(random.randint(2, 4)):
        r = params['radius'] * (0.3 + 0.7 * i / 4)
        msp.add_circle((offset_x, 0), r)
    
    # 键槽
    if random.random() > 0.3:
        slot_w = params['radius'] * 0.3
        slot_d = params['radius'] * 0.15
        slot_len = params['length'] * 0.3
        msp.add_line((-slot_len/2, params['radius']*0.8 - slot_d), 
                     (slot_len/2, params['radius']*0.8 - slot_d))
        msp.add_line((-slot_len/2, params['radius']*0.8), 
                     (slot_len/2, params['radius']*0.8))

def create_gear(doc, msp, params):
    """生成传动件图纸 - 齿轮、链轮"""
    cx, cy = 0, 0
    
    # 齿数
    teeth = random.randint(12, 36)
    r_outer = params['radius']
    r_inner = r_outer * 0.85
    r_root = r_outer * 0.75
    
    # 外圆
    msp.add_circle((cx, cy), r_outer)
    
    # 齿根圆
    msp.add_circle((cx, cy), r_root)
    
    # 内孔
    msp.add_circle((cx, cy), params['bore'])
    
    # 绘制齿形
    for i in range(teeth):
        angle = 2 * math.pi * i / teeth
        angle2 = 2 * math.pi * (i + 0.3) / teeth
        angle3 = 2 * math.pi * (i + 0.7) / teeth
        
        # 齿顶
        x1 = cx + r_outer * math.cos(angle)
        y1 = cy + r_outer * math.sin(angle)
        x2 = cx + r_outer * math.cos(angle2)
        y2 = cy + r_outer * math.sin(angle2)
        
        # 齿根
        x3 = cx + r_root * math.cos(angle2)
        y3 = cy + r_root * math.sin(angle2)
        x4 = cx + r_root * math.cos(angle3)
        y4 = cy + r_root * math.sin(angle3)
        
        msp.add_line((x1, y1), (x2, y2))
        msp.add_line((x2, y2), (x3, y3))
        msp.add_line((x3, y3), (x4, y4))
    
    # 键槽
    slot_w = params['bore'] * 0.4
    slot_d = params['bore'] * 0.15
    msp.add_line((-slot_w/2, params['bore'] + slot_d), (slot_w/2, params['bore'] + slot_d))
    msp.add_line((-slot_w/2, params['bore']), (-slot_w/2, params['bore'] + slot_d))
    msp.add_line((slot_w/2, params['bore']), (slot_w/2, params['bore'] + slot_d))
    
    # 中心线
    msp.add_line((-r_outer - 10, 0), (r_outer + 10, 0), dxfattribs={'layer': 'CENTER'})
    msp.add_line((0, -r_outer - 10), (0, r_outer + 10), dxfattribs={'layer': 'CENTER'})

def create_housing(doc, msp, params):
    """生成壳体类图纸 - 箱体、端盖、支座"""
    w, h = params['width'], params['height']
    
    # 主轮廓
    msp.add_line((-w/2, -h/2), (w/2, -h/2))
    msp.add_line((w/2, -h/2), (w/2, h/2))
    msp.add_line((w/2, h/2), (-w/2, h/2))
    msp.add_line((-w/2, h/2), (-w/2, -h/2))
    
    # 内腔
    inner_w = w * 0.7
    inner_h = h * 0.7
    msp.add_line((-inner_w/2, -inner_h/2), (inner_w/2, -inner_h/2))
    msp.add_line((inner_w/2, -inner_h/2), (inner_w/2, inner_h/2))
    msp.add_line((inner_w/2, inner_h/2), (-inner_w/2, inner_h/2))
    msp.add_line((-inner_w/2, inner_h/2), (-inner_w/2, -inner_h/2))
    
    # 安装孔 (四角)
    hole_r = min(w, h) * 0.05
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        hx = dx * (w/2 - w*0.1)
        hy = dy * (h/2 - h*0.1)
        msp.add_circle((hx, hy), hole_r)
    
    # 中心轴承孔
    if random.random() > 0.3:
        bearing_r = min(w, h) * 0.15
        msp.add_circle((0, 0), bearing_r)
        msp.add_circle((0, 0), bearing_r * 0.6)
    
    # 法兰
    if random.random() > 0.5:
        flange_w = w * 1.2
        flange_h = h * 0.15
        y_offset = -h/2 - flange_h/2
        msp.add_line((-flange_w/2, y_offset - flange_h/2), (flange_w/2, y_offset - flange_h/2))
        msp.add_line((-flange_w/2, y_offset + flange_h/2), (flange_w/2, y_offset + flange_h/2))
        msp.add_line((-flange_w/2, y_offset - flange_h/2), (-flange_w/2, y_offset + flange_h/2))
        msp.add_line((flange_w/2, y_offset - flange_h/2), (flange_w/2, y_offset + flange_h/2))
        
        # 法兰孔
        for dx in [-0.8, -0.4, 0.4, 0.8]:
            msp.add_circle((dx * flange_w/2, y_offset), hole_r * 0.8)

def create_connector(doc, msp, params):
    """生成连接件图纸 - 螺栓、螺母、销"""
    part_type = random.choice(['bolt', 'nut', 'pin', 'washer'])
    
    if part_type == 'bolt':
        # 螺栓
        head_r = params['size'] * 0.8
        shaft_r = params['size'] * 0.4
        length = params['size'] * 3
        
        # 六角头 (简化为圆)
        msp.add_circle((0, length/2 + head_r), head_r)
        
        # 杆身
        msp.add_line((-shaft_r, -length/2), (-shaft_r, length/2))
        msp.add_line((shaft_r, -length/2), (shaft_r, length/2))
        
        # 螺纹线
        thread_start = -length/2
        thread_end = length/4
        for y in range(int(thread_start), int(thread_end), int(params['size']*0.3)):
            msp.add_line((-shaft_r, y), (shaft_r, y + params['size']*0.15))
        
    elif part_type == 'nut':
        # 螺母
        outer_r = params['size']
        inner_r = params['size'] * 0.5
        
        # 外六边形
        for i in range(6):
            a1 = math.pi * i / 3
            a2 = math.pi * (i + 1) / 3
            msp.add_line((outer_r * math.cos(a1), outer_r * math.sin(a1)),
                        (outer_r * math.cos(a2), outer_r * math.sin(a2)))
        
        # 内孔
        msp.add_circle((0, 0), inner_r)
        
        # 螺纹
        for i in range(3):
            r = inner_r + (outer_r - inner_r) * 0.1 * i
            msp.add_circle((0, 0), r, dxfattribs={'linetype': 'DASHED'})
            
    elif part_type == 'pin':
        # 销钉
        r = params['size'] * 0.3
        length = params['size'] * 2
        
        msp.add_line((-r, -length/2), (-r, length/2))
        msp.add_line((r, -length/2), (r, length/2))
        msp.add_arc((0, -length/2), r, 180, 360)
        msp.add_arc((0, length/2), r, 0, 180)
        
    else:  # washer
        # 垫圈
        outer_r = params['size']
        inner_r = params['size'] * 0.5
        msp.add_circle((0, 0), outer_r)
        msp.add_circle((0, 0), inner_r)

def create_other(doc, msp, params):
    """生成其他类图纸 - 混合特征"""
    # 随机组合多种元素
    elements = random.randint(3, 8)
    
    for _ in range(elements):
        elem_type = random.choice(['line', 'circle', 'arc', 'rect'])
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        
        if elem_type == 'line':
            dx = random.uniform(10, 50)
            dy = random.uniform(10, 50)
            msp.add_line((x, y), (x + dx, y + dy))
        elif elem_type == 'circle':
            r = random.uniform(5, 30)
            msp.add_circle((x, y), r)
        elif elem_type == 'arc':
            r = random.uniform(10, 40)
            a1 = random.uniform(0, 180)
            a2 = a1 + random.uniform(30, 180)
            msp.add_arc((x, y), r, a1, a2)
        else:  # rect
            w = random.uniform(20, 60)
            h = random.uniform(20, 60)
            msp.add_line((x, y), (x + w, y))
            msp.add_line((x + w, y), (x + w, y + h))
            msp.add_line((x + w, y + h), (x, y + h))
            msp.add_line((x, y + h), (x, y))
    
    # 添加一些文字
    if random.random() > 0.5:
        msp.add_text("NOTE", dxfattribs={'height': 5, 'layer': 'TEXT'})

def generate_samples(category: str, count: int, output_dir: Path):
    """为指定类别生成样本"""
    cat_dir = output_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    generators = {
        '轴类': (create_shaft, lambda: {'length': random.uniform(100, 300), 'radius': random.uniform(15, 50)}),
        '传动件': (create_gear, lambda: {'radius': random.uniform(30, 80), 'bore': random.uniform(8, 20)}),
        '壳体类': (create_housing, lambda: {'width': random.uniform(80, 200), 'height': random.uniform(60, 150)}),
        '连接件': (create_connector, lambda: {'size': random.uniform(10, 30)}),
        '其他': (create_other, lambda: {}),
    }
    
    gen_func, param_func = generators[category]
    
    for i in range(count):
        doc = ezdxf.new('R2010')
        doc.units = units.MM
        
        # 创建图层
        doc.layers.add('CENTER', color=1)
        doc.layers.add('TEXT', color=3)
        doc.layers.add('DIM', color=2)
        
        msp = doc.modelspace()
        params = param_func()
        
        try:
            gen_func(doc, msp, params)
            filename = cat_dir / f"syn_{i+1:04d}.dxf"
            doc.saveas(filename)
        except Exception as e:
            print(f"  生成失败 {category}/{i}: {e}")
            continue
    
    print(f"  {category}: 生成 {count} 个文件")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== 合成DXF训练数据 ===\n")
    
    # 每类生成数量 (重点补充少数类)
    counts = {
        '轴类': 200,
        '传动件': 300,  # 最少的类，多生成
        '壳体类': 200,
        '连接件': 200,
        '其他': 100,
    }
    
    for category, count in counts.items():
        generate_samples(category, count, OUTPUT_DIR)
    
    # 创建manifest
    import json
    manifest = []
    label_map = {'传动件': 0, '其他': 1, '壳体类': 2, '轴类': 3, '连接件': 4}
    
    for cat, label_id in label_map.items():
        cat_dir = OUTPUT_DIR / cat
        if cat_dir.exists():
            for f in cat_dir.glob('*.dxf'):
                manifest.append({
                    'file': f"{cat}/{f.name}",
                    'category': cat,
                    'label_id': label_id,
                    'source': 'synthetic'
                })
    
    with open(OUTPUT_DIR / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / 'labels.json', 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_map,
            'id_to_label': {v: k for k, v in label_map.items()},
            'version': 'synthetic'
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n总计生成 {len(manifest)} 个合成样本")
    print(f"保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
