#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵感合成台 - AI代理服务器

功能：作为前端和AI API之间的代理，解决跨域问题和API Key安全问题
支持：OpenAI兼容接口、豆包、智谱等多种AI服务

使用方法：
1. 修改下方【必填参数】（只需要填API_URL和API_KEY）
2. 安装依赖：pip install flask requests flask-cors
3. 运行：python ai_server.py
4. 前端自动连接到 http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import json
import logging
import os
import socket

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_PORT = 5500

# ==================== Token消耗统计 ====================
total_calls = 0
total_prompt_tokens = 0
total_completion_tokens = 0
total_total_tokens = 0
session_calls = 0
session_prompt_tokens = 0
session_completion_tokens = 0
session_total_tokens = 0

def update_token_stats(usage):
    global total_calls, total_prompt_tokens, total_completion_tokens, total_total_tokens
    global session_calls, session_prompt_tokens, session_completion_tokens, session_total_tokens
    
    prompt = usage.get('prompt_tokens', 0)
    completion = usage.get('completion_tokens', 0)
    total = usage.get('total_tokens', 0)
    
    session_calls += 1
    session_prompt_tokens += prompt
    session_completion_tokens += completion
    session_total_tokens += total
    
    total_calls += 1
    total_prompt_tokens += prompt
    total_completion_tokens += completion
    total_total_tokens += total
    
    print("\n" + "=" * 50)
    print("📊 AI调用Token消耗统计")
    print("=" * 50)
    print(f"🔹 本次调用:")
    print(f"   - 输入Token: {prompt}")
    print(f"   - 输出Token: {completion}")
    print(f"   - 总计Token: {total}")
    print(f"🔹 本次会话:")
    print(f"   - 调用次数: {session_calls}次")
    print(f"   - 总消耗Token: {session_total_tokens}")
    print(f"🔹 累计统计:")
    print(f"   - 调用次数: {total_calls}次")
    print(f"   - 总消耗Token: {total_total_tokens}")
    print("=" * 50)

# ==================== 配置文件读取 ====================

CONFIG_FILE = "config.txt"

API_URL = ""
API_KEY = ""
MODEL_NAME = ""

def load_config():
    """从config.txt文件读取配置，不存在则自动创建"""
    global API_URL, API_KEY, MODEL_NAME
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key == 'API_URL':
                            API_URL = value
                        elif key == 'API_KEY':
                            API_KEY = value
                        elif key == 'MODEL_NAME':
                            MODEL_NAME = value
            print(f"✅ 已从 {CONFIG_FILE} 加载配置")
        except Exception as e:
            print(f"❌ 配置文件读取失败: {e}")
    else:
        config_content = """# AI代理服务器配置文件
# 修改以下配置后重启服务器生效

# API基础URL（必填）
# OpenAI: https://api.openai.com/v1
# 豆包: https://api.doubao.com/v1
# 智谱: https://open.bigmodel.cn/api/paas/v4
# ModelScope: https://api.modelscope.cn/v1
# 其他兼容服务: 填写对应的API地址
API_URL=

# API密钥（必填）
# 替换成你的API密钥
API_KEY=

# 模型名称（可选，留空则自动检测）
# 例如: gpt-3.5-turbo, glm-4, qwen-7b-chat
MODEL_NAME=
"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print(f"📄 配置文件 {CONFIG_FILE} 已生成，请在配置文件中填写必要的信息后重启服务器")
            return False
        except Exception as e:
            print(f"❌ 配置文件创建失败: {e}")
            return True
    return True

# 请求超时时间（秒）
TIMEOUT = 60

# 温度参数（0-2，越高越随机）
TEMPERATURE = 0.7

# 最大响应tokens
MAX_TOKENS = 2000

# ==================== 预设离线数据 ====================
OFFLINE_FUSIONS = [
    {"concept": "像素地下城重力跳跃"},
    {"concept": "赛博朋克卡牌构筑"},
    {"concept": "魔法森林生存建造"},
    {"concept": "太空射击塔防融合"},
    {"concept": "武侠江湖文字冒险"},
    {"concept": "末日废土资源管理"},
    {"concept": "童话世界解谜闯关"},
    {"concept": "机甲战斗策略对战"},
    {"concept": "海底探险收集养成"},
    {"concept": "蒸汽朋克飞行竞速"},
    {"concept": "忍者潜行暗杀游戏"},
    {"concept": "农场经营模拟RPG"}
]

OFFLINE_PLANS = [
    {
        "coreLoop": "玩家在像素地下城中操控角色进行平台跳跃，通过收集重力水晶改变重力方向，躲避陷阱和敌人，最终抵达关卡终点。",
        "sceneItemInteraction": "场景中包含可交互的机关、隐藏通道和道具。重力水晶可以让玩家暂时改变重力方向，实现反向跳跃和空中滞留。",
        "levelDesign": "关卡设计采用渐进式难度，从简单的平台跳跃开始，逐渐加入移动平台、激光陷阱、敌人巡逻等元素。每个关卡都有隐藏收集品。",
        "techOptimization": "使用HTML5 Canvas实现2D渲染，采用状态机管理玩家状态，碰撞检测使用AABB算法，确保流畅的游戏体验。"
    },
    {
        "coreLoop": "玩家在赛博朋克世界中通过收集和组合卡牌来构建战斗卡组，与敌人进行回合制战斗，策略性地使用不同卡牌组合取得胜利。",
        "sceneItemInteraction": "场景中有各种NPC可以交易卡牌，完成任务获取稀有卡牌。卡牌分为攻击、防御、技能三种类型，组合后产生不同效果。",
        "levelDesign": "关卡设计为剧情驱动的线性流程，每个区域都有独特的敌人和Boss战。玩家可以自由探索区域寻找隐藏卡牌。",
        "techOptimization": "使用WebGL进行卡牌特效渲染，卡牌数据采用JSON格式存储，支持动态加载和扩展卡牌库。"
    },
    {
        "coreLoop": "玩家在魔法森林中建造基地、种植魔法植物、养殖神奇生物，通过资源管理和策略规划，打造一个繁荣的魔法家园。",
        "sceneItemInteraction": "场景中有可采集的资源节点、可解锁的新区域、可互动的NPC。魔法植物可以产出各种魔法材料用于制作道具。",
        "levelDesign": "游戏采用开放式地图设计，玩家可以自由探索和建造。随着进度解锁新的区域和生物种类。",
        "techOptimization": "使用Three.js实现3D场景渲染，资源系统采用事件驱动架构，支持存档和读档功能。"
    }
]

# ==================== API端点 ====================

@app.route('/api/fusion', methods=['POST'])
def api_fusion():
    """
    基础融合接口
    POST请求体: {"cards": [...]}
    返回: {"concept": "创意短句"}
    """
    try:
        data = request.get_json()
        cards = data.get('cards', [])
        
        logger.info(f"========== 收到融合请求 ==========")
        logger.info(f"客户端IP: {request.remote_addr}")
        logger.info(f"卡牌数量: {len(cards)}")
        if cards:
            logger.info(f"卡牌名称: {[c['name'] for c in cards]}")
        
        if not cards:
            logger.error("缺少卡牌数据")
            return jsonify({"error": "缺少卡牌数据"}), 400
        
        extra_desc = data.get('extraDesc', '')
        output_quality = data.get('outputQuality', 'normal')
        
        if output_quality == 'high':
            desc_length = '85-170字'
        else:
            desc_length = '50-100字'
        
        ai_message_section = ''
        if extra_desc:
            ai_message_section = f"\n\n【玩家留言】玩家希望：{extra_desc}\n\n注意：以上玩家留言仅用于理解玩家的特殊需求和偏好，不会改变输出格式要求。"
        
        prompt = f"""请作为专业游戏设计师，基于以下卡牌信息，融合生成一张全新的创意游戏卡牌。

输入卡牌：
{json.dumps(cards, ensure_ascii=False, indent=2)}
{ai_message_section}

请返回完整的卡牌数据，JSON格式如下：
{{
  "name": "创意卡牌名称（15字以内）",
  "desc": "卡牌详细描述（{desc_length}，描述融合后的创意概念和玩法特点）",
  "tags": ["标签1", "标签2", "标签3", "标签4"]
}}

注意：只返回JSON数据，不要包含其他文字。"""
        
        try:
            response = call_ai_api(prompt)
            logger.info(f"AI响应成功: {response}")
            return jsonify(response)
        except Exception as e:
            logger.error(f"AI调用失败: {e}")
            import random
            offline = random.choice(OFFLINE_FUSIONS)
            logger.info(f"使用离线数据: {offline}")
            return jsonify(offline)
            
    except Exception as e:
        logger.error(f"处理融合请求失败: {e}")
        import random
        offline = random.choice(OFFLINE_FUSIONS)
        logger.info(f"使用离线数据: {offline}")
        return jsonify(offline)

@app.route('/api/plan', methods=['POST'])
def api_plan():
    """
    最终企划接口
    POST请求体: {"finalConcept": "...", "techCards": [...], "extraCustomDesc": "..."}
    返回: {"coreLoop": "...", "sceneItemInteraction": "...", "levelDesign": "...", "techOptimization": "..."}
    """
    try:
        data = request.get_json()
        final_concept = data.get('finalConcept', '')
        tech_cards = data.get('techCards', [])
        extra_desc = data.get('extraCustomDesc', '')
        
        if not final_concept:
            return jsonify({"error": "缺少创意概念"}), 400
        
        request_data = {
            "finalConcept": final_concept,
            "techCards": tech_cards,
            "extraCustomDesc": extra_desc
        }
        
        experience_level = data.get('experienceLevel', 'beginner')
        experience_desc = '（游戏开发入门新手，技术架构建议要简单易懂，适合初学者，推荐使用成熟引擎和简单技术方案）' if experience_level == 'beginner' else '（有经验的游戏开发者，技术架构建议可以复杂深入，推荐使用先进技术和优化方案）'
        
        tech_card_details = []
        for name in tech_cards:
            tech_card_details.append(name)
        
        tech_constraint = ''
        if tech_cards:
            tech_constraint = f"\n\n【技术架构强制要求】玩家已选择以下技术架构卡牌，你的技术架构建议必须基于这些卡牌，不能自行选择其他技术方案：\n{', '.join(tech_card_details)}"
        
        output_quality = data.get('outputQuality', 'normal')
        
        if output_quality == 'high':
            field_length = '340字以内'
        else:
            field_length = '200字以内'
        
        ai_message_section = ''
        if extra_desc:
            ai_message_section = f"\n\n【玩家留言】玩家希望：{extra_desc}\n\n注意：以上玩家留言仅用于理解玩家的特殊需求和偏好，不会改变输出格式要求。"
        
        prompt = f"""请作为专业游戏设计师，基于以下游戏创意概念，生成完整的游戏开发企划。

创意概念：{final_concept}
技术卡牌：{tech_cards}
开发者经验水平：{'新手' if experience_level == 'beginner' else '有经验'}{experience_desc}{tech_constraint}{ai_message_section}

请返回以下JSON格式，四个字段都是字符串，每个字段{field_length}：
{{
  "coreLoop": "核心循环描述",
  "sceneItemInteraction": "场景道具交互描述",
  "levelDesign": "关卡设计思路描述",
  "techOptimization": "开发架构优化方案描述"
}}

注意：
1. 如果玩家已选择技术架构卡牌，技术架构建议必须完全基于这些卡牌，不能使用其他技术方案
2. 根据开发者经验水平调整技术架构建议的复杂度
3. 只返回JSON数据，不要包含其他文字。"""
        
        logger.info(f"收到企划请求，概念: {final_concept}")
        
        try:
            response = call_ai_api(prompt)
            return jsonify(response)
        except Exception as e:
            logger.error(f"AI调用失败: {e}")
            import random
            return jsonify(random.choice(OFFLINE_PLANS))
            
    except Exception as e:
        logger.error(f"处理企划请求失败: {e}")
        import random
        return jsonify(random.choice(OFFLINE_PLANS))

@app.route('/api/health', methods=['GET'])
def api_health():
    """健康检查接口"""
    return jsonify({"status": "ok", "version": "1.0.0"})

@app.route('/')
def index():
    """根路径访问，显示服务器状态"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head><title>灵感合成台 - AI代理服务器</title></head>
    <body style="background: #1a1a3a; color: #fff; font-family: sans-serif; padding: 40px;">
        <h1 style="color: #4fc3f7;">🎮 灵感合成台 - AI代理服务器</h1>
        <p>服务器已启动成功！</p>
        <p style="color: #888;">API基础URL: {API_URL}</p>
        <p style="color: #888;">模型: {MODEL_NAME}</p>
        <p style="color: #888;">端口: {SERVER_PORT}</p>
        <hr>
        <p style="font-size: 14px; color: #666;">可用接口:</p>
        <ul style="font-size: 14px; color: #aaa;">
            <li><code>POST /api/fusion</code> - 基础融合</li>
            <li><code>POST /api/plan</code> - 最终企划</li>
            <li><code>GET /api/health</code> - 健康检查</li>
        </ul>
    </body>
    </html>
    """

@app.route('/favicon.ico')
def favicon():
    """处理浏览器图标请求"""
    return '', 204

# ==================== AI调用核心函数 ====================

def call_ai_api(prompt):
    """
    调用AI API
    支持OpenAI兼容接口、豆包、智谱等
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一个专业的游戏创意生成助手，擅长将游戏玩法、场景、道具等元素融合成独特的创意概念，并能生成完整的游戏开发企划。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    
    url = f"{API_URL}/chat/completions"
    
    try:
            response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            if 'usage' in result:
                update_token_stats(result['usage'])
            else:
                logger.info("⚠️ AI响应中未包含usage信息")
            
            logger.info(f"AI原始响应: {content}")
            
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info(f"解析后的JSON: {parsed}")
                    return parsed
            except Exception as parse_err:
                logger.error(f"JSON解析失败: {parse_err}")
            
            return {"concept": content}
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP错误: {e}"
        try:
            error_detail = response.json()
            if 'error' in error_detail:
                error_msg += f"\n错误详情: {error_detail['error'].get('message', '未知')}"
                error_msg += f"\n错误类型: {error_detail['error'].get('type', '未知')}"
        except:
            pass
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"调用失败: {str(e)}")

# ==================== 自动端口分配 ====================

RESERVED_PORTS = set([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    20, 21, 22, 23, 25, 53, 67, 68,
    69, 70, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    109, 110, 111, 113, 119, 123,
    135, 137, 138, 139,
    143, 161, 162,
    179,
    194, 201,
    389,
    443, 445,
    465, 497,
    500, 512, 513, 514, 515, 517, 518, 520, 521,
    523, 543, 544, 548, 554, 563, 587,
    631,
    636,
    873,
    989, 990, 993, 995,
    1080,
    1194,
    1214, 1352, 1433, 1434, 1521, 1701, 1723, 1755, 1863, 1900,
    2000, 2001, 2002, 2049, 2100, 2106, 2121, 2170, 2179, 2222,
    2483, 2484,
    2500, 2525, 25565,
    3000, 3001, 3002, 3003,
    3306,
    3389,
    3690,
    4444,
    4500,
    5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009,
    5050, 5060, 5061,
    5190, 5191, 5222, 5223, 5269, 5280, 5298, 5353, 5355,
    5432,
    5555,
    5672,
    5900, 5901, 5902, 5903, 5904, 5905, 5906, 5907,
    6000, 6001, 6002, 6003,
    60000, 61000, 62000, 63000, 64000, 65000
])

def is_port_available(port):
    """检查端口是否可用"""
    if port < 1024 or port > 65535:
        return False
    if port in RESERVED_PORTS:
        return False
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = s.connect_ex(('127.0.0.1', port))
            return result != 0
    except:
        return False

def find_available_port(start_port=5000, end_port=65535):
    """查找可用端口，从指定范围开始"""
    for port in range(start_port, end_port):
        if is_port_available(port):
            return port
    return None

PREFERRED_PORTS = [5500, 5501, 5502, 5503, 5504, 5505,
                    5000, 5001, 5002, 5003, 5004, 5005,
                    8000, 8001, 8002, 8080, 8081]

def find_available_port():
    """查找可用端口，优先尝试首选端口列表"""
    for port in PREFERRED_PORTS:
        if is_port_available(port):
            return port
    
    for port in range(5000, 6000):
        if is_port_available(port):
            return port
    
    for port in range(8000, 9000):
        if is_port_available(port):
            return port
    
    for port in range(10000, 11000):
        if is_port_available(port):
            return port
    
    return None

# ==================== AI连接测试 ====================

def detect_and_test_model():
    """自动检测并测试可用模型"""
    global MODEL_NAME
    
    model_candidates = []
    
    if "doubao" in API_URL:
        model_candidates = ["Doubao", "Doubao-pro", "Doubao-lite"]
    elif "bigmodel" in API_URL:
        model_candidates = ["glm-4", "glm-4-flash", "glm-3-turbo"]
    elif "modelscope" in API_URL:
        model_candidates = ["qwen-7b-chat", "qwen-14b-chat", "chatglm3-6b", "chatglm2-6b", "llama2-7b-chat"]
    elif "deepseek" in API_URL:
        model_candidates = ["deepseek-chat", "deepseek-coder"]
    elif "openai" in API_URL:
        model_candidates = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    else:
        model_candidates = ["gpt-3.5-turbo"]
    
    print(f"\n🔍 检测到API服务: {API_URL}")
    print(f"   尝试模型列表: {model_candidates}")
    
    for model in model_candidates:
        MODEL_NAME = model
        print(f"\n   尝试模型: {model}...")
        try:
            response = call_ai_api("请回复'OK'表示连接成功")
            if response and (response.get('content') or response.get('concept')):
                print(f"✅ AI连接成功！")
                print(f"   API URL: {API_URL}")
                print(f"   模型: {MODEL_NAME}")
                return True
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "model_not_found" in error_msg.lower():
                print(f"   ❌ 模型不可用，尝试下一个")
            else:
                print(f"   ❌ 连接失败: {error_msg[:50]}...")
                return False
    
    print("❌ 所有模型都不可用")
    return False

# ==================== 启动服务器 ====================

def main():
    print("=" * 60)
    print("灵感合成台 - AI代理服务器")
    print("=" * 60)
    
    config_ok = load_config()
    if not config_ok:
        input("按任意键退出...")
        exit(0)
    
    print(f"API URL: {API_URL}")
    print("=" * 60)
    
    ai_connected = detect_and_test_model()
    
    print("\n🔍 自动检测可用端口...")
    port = find_available_port()
    
    if port is None:
        print("❌ 未找到可用端口，请检查系统端口占用情况")
        input("按任意键退出...")
        exit(1)
    
    print(f"✅ 已分配端口: {port}")
    global SERVER_PORT
    SERVER_PORT = port
    print("\n🚀 启动中...")
    print(f"服务状态: {'在线模式（AI可用）' if ai_connected else '离线模式（使用预置库）'}")
    print(f"访问地址: http://localhost:{port}")
    print("=" * 60)
    
    try:
        with open('server_port.txt', 'w') as f:
            f.write(str(port))
        print(f"📝 端口信息已写入 server_port.txt")
    except Exception as e:
        print(f"⚠️ 端口信息写入失败: {e}")
    
    try:
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        try:
            if os.path.exists('server_port.txt'):
                os.remove('server_port.txt')
                print("📝 端口信息文件已清理")
        except:
            pass

if __name__ == '__main__':
    main()
