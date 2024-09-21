import gradio as gr
from groq import Groq
import json
from datetime import datetime

import os

API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=API_KEY)


def write_string_to_json(input_string):
    # Get current datetime
    current_time = datetime.now()

    # Format datetime for filename
    filename_time = current_time.strftime("%Y%m%d_%H%M%S")

    # Create filename using the datetime
    filename = f"{filename_time}.json"

    # Create a dictionary with the input string and ISO formatted datetime
    data = {"content": input_string, "timestamp": current_time.isoformat()}

    # Write the dictionary to a JSON file with UTF-8 encoding
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"String and timestamp written to {filename}")


def generate_assessment(
    age,
    gender,
    occupation,
    activity_level,
    other_comments,
    pain_location,
    pain_intensity,
    pain_duration,
    pain_character,
    aggravating_factors,
    relieving_factors,
    previous_injuries,
    chronic_conditions,
    medications,
    functional_limitations,
    patient_goals,
    equipment_at_home,
):

    prompt = f"""
        任务：
        你是一位专门从事物理治疗评估的AI助手。你的任务是使用提供的信息为我进行初步评估。分析数据并生成一个简明的总结，包括我的病情、可能的诊断以及进一步评估或治疗的建议。
        Copy使用以下信息进行你的评估：

        1. 我的人口统计数据：
        年龄：{age}
        性别：{gender}
        职业：{occupation}
        活动水平：{activity_level}

        2. 我的疼痛信息：
        部位：{pain_location}
        强度（0-10分制）：{pain_intensity}
        持续时间：{pain_duration}
        特征（例如，尖锐、钝痛、酸痛）：{pain_character}
        加重因素：{aggravating_factors}
        缓解因素：{relieving_factors}
        其他评论：{other_comments}

        3. 我的病史：
        既往伤病：{previous_injuries}
        慢性病：{chronic_conditions}
        药物：{medications}

        4. 个性化和目标：
        我的功能限制：{functional_limitations}
        我的目标：{patient_goals}
        拥有的设备：{equipment_at_home}

        评估报告：

        根据以上信息，生成一份评估报告，包括以下内容：

        1. 病情总结（2-3句话）
        2. 任何需要立即医疗关注的红旗警示或担忧（如果适用）。如果不适用，请说明没有即时的红旗警示或担忧。
        3. 诊断可能性表格，列出诊断及其概率（高、中、低）（列出2-3种可能性）。
        4. 可以在家进行的进一步评估建议，以及如何解读评估结果（3-4点）。
        5. 根据个性化和目标部分，建议4-5个个性化治疗计划，如推荐的运动、康复活动。如果适用，建议重复次数和重量。按最推荐程度（从高到低）排列活动。

        以清晰、简洁的方式呈现你的评估，适合我和物理治疗师审阅。记住保持专业的语气，并强调这是基于提供信息的初步评估，而不是最终诊断。

        使用格式良好的Markdown生成文本，在必要时使用粗体和表格使评估更清晰。

        回答：
    """

    write_string_to_json(prompt)

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=0.95,
        stream=True,
        stop=None,
    )

    output = []
    for chunk in completion:
        text = chunk.choices[0].delta.content or ""
        output.append(text)
        yield "".join(output)


# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 个性化康复治疗评估")

    # with gr.Row():
    # with gr.Column():
    age = gr.Number(label="年龄")
    gender = gr.Radio(["男", "女", "其他"], label="性别")
    occupation = gr.Text(
        label="职业",
        info="例如：办公室职员、建筑工人、学生",
    )
    activity_level = gr.Radio(["久坐", "低度", "中度", "高度"], label="运动量")

    # with gr.Column():
    pain_location = gr.Text(
        label="疼痛部位",
        info="例如：下背部、右膝、左肩",
    )
    pain_intensity = gr.Slider(
        0,
        10,
        step=1,
        label="疼痛强度（0-10）",
        info="0 = 无痛，10 = 最强烈疼痛",
    )
    pain_duration = gr.Radio(
        [
            "最近",
            "不足1个月",
            "1至3个月",
            "3至6个月",
            "1年或更长",
        ],
        label="疼痛持续时间",
        info="您已经痛了多长时间？",
    )
    pain_character = gr.CheckboxGroup(
        [
            "灼烧感",
            "酸痛",
            "刺痛",
            "跳痛",
            "麻木",
            "刺痒",
            "放射痛",
        ],
        label="疼痛特征",
        info="选择所有适用项",
    )
    other_comments = gr.Text(label="还有什么需要补充的吗？")

    # with gr.Row():
    # with gr.Column():
    aggravating_factors = gr.Text(
        label="加重疼痛的活动",
        info="例如：长时间坐着、上楼梯",
    )
    relieving_factors = gr.CheckboxGroup(
        [
            "冰敷或热敷",
            "服用非处方止痛药",
            "练习放松技巧（如深呼吸、冥想）",
            "轻柔拉伸或瑜伽",
            "按摩受影响区域",
            "休息并避免加重活动",
            "使用局部止痛膏或凝胶",
        ],
        label="缓解疼痛的方式",
    )
    previous_injuries = gr.Text(
        label="以前有过相关伤病吗？",
        info="例如：两年前扭伤脚踝",
    )

    # with gr.Column():
    chronic_conditions = gr.Text(label="慢性病", info="例如：糖尿病、高血压")
    medications = gr.Text(
        label="药物",
        info="例如：需要时服用布洛芬、血压药",
    )
    functional_limitations = gr.Text(
        label="功能限制",
        info="例如：无法坐超过30分钟、难以下车",
    )
    equipment_at_home = gr.CheckboxGroup(
        [
            "哑铃",
            "弹力带",
            "瑜伽垫",
            "健身球",
            "跳绳",
            "跑步机",
            "固定自行车",
            "壶铃",
            "引体向上杆",
            "泡沫轴",
            "筋膜枪",
        ],
        label="家中现有设备",
        info="选择所有适用项",
    )
    patient_goals = gr.Text(
        label="我的目标",
        info="例如：每周打两次羽毛球而不感到疼痛",
    )

    submit_btn = gr.Button("生成评估")
    output = gr.Markdown(label="生成的评估")

    submit_btn.click(
        generate_assessment,
        inputs=[
            age,
            gender,
            occupation,
            activity_level,
            other_comments,
            pain_location,
            pain_intensity,
            pain_duration,
            pain_character,
            aggravating_factors,
            relieving_factors,
            previous_injuries,
            chronic_conditions,
            medications,
            functional_limitations,
            patient_goals,
            equipment_at_home,
        ],
        outputs=output,
    )

# Launch the Gradio interface
if __name__ == "__main__":
    demo.queue(max_size=20).launch(server_name="0.0.0.0")
