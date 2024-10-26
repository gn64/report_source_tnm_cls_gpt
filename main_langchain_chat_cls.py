import os
import logging
import sys
import pandas as pd
import pickle
import re
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def get_csv(input_path):
    df = pd.read_csv(input_path)
    input_dic = []
    for i in range(len(df)):
        tmp_dic = {}
        tmp_dic["ID"] = df.loc[i, "ID"]
        tmp_dic["T"] = df.loc[i, "T"]
        tmp_dic["N"] = df.loc[i, "N"]
        tmp_dic["M"] = df.loc[i, "M"]
        with open("../../dataset/" + str(df.loc[i, "ID"]) + ".txt", "r") as f:
            tmp_str = f.read()
            tmp_dic["Report"] = tmp_str
        if i < 5:
            input_dic.append(tmp_dic)
        else:
            input_dic.append(tmp_dic)
    return input_dic

def process_tnm(process_dic, prefix=""):
    os.environ["OPENAI_API_KEY"] = ""
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory)
    # TNM定義ファイルの読み込み
    with open("../../init_data/tnm_en.txt", "r") as f:
        docu_text = f.read()
    first_input = (
        "あなたは熟練の呼吸器外科医です。\
            以下のTNMの定義を元にして後に与えられる読影レポートのTNM分類を作成してください。\
            TNMの定義:"
        + docu_text
        + "\n所見に記載が無い場合には異常な所見が無かったものと考えてください。"
    )
    conversation.predict(input=first_input)
    first_pickel = pickle.dumps(conversation.memory)
    # TNM classification prompt
    classification_prompt = PromptTemplate(
        input_variables=["report"],
        template="以下の読影レポートを理解してTNM分類を出力してください。\
            当てはまるものがない場合には空にしてください。\
            出力は正確に次の形式に従ってください： | T:value,N:value,M:value |\
                余分な情報や詳細な説明は一切必要ありません。直接TNM分類の値だけを出力してください\ｎ\
                Report:{report}",
    )
    classification_chain = LLMChain(llm=llm, prompt=classification_prompt)
    result_list = []
    for tmp_dic in process_dic:
        conv = ConversationChain(llm=llm, memory=pickle.loads(first_pickel))
        input_str = (
            "以下の読影レポートを理解してTNM分類を出力してください。\
            当てはまるものがない場合には空にしてください。\
            出力は正確に次の形式に従ってください： {T:value,N:value,M:value}。\
                余分な情報や詳細な説明は一切必要ありません。直接TNM分類の値だけを出力してください\ｎ\
                Report:"
            + tmp_dic["Report"]
        )
        result_str_1 = conv.predict(input=input_str)
        result_str_2 = classification_chain.run(result_str_1)
        tmp_dic_result = tmp_dic.copy()
        tmp_dic_result["result"] = result_str_1
        tmp_dic_result["classification"] = result_str_2
        result_list.append(tmp_dic_result)
        pattern = r" T(\S+) *, *N *(\S+) *, *M *(\S+) * "
        tmp_dic_result["error_str"] = ""
        match = re.search(pattern, result_str_2)
        try:
            label_t = match.group(1)
        except:
            label_t = "error"
            tmp_dic_result["error_str"] = result_str_2
        try:
            label_n = match.group(2)
        except:
            label_n = "error"
            tmp_dic_result["error_str"] = result_str_2
        try:
            label_m = match.group(3)
        except:
            label_m = "error"
            tmp_dic_result["error_str"] = result_str_2
        tmp_dic_result[f"{prefix}_t"] = label_t
        tmp_dic_result[f"{prefix}_n"] = label_n
        tmp_dic_result[f"{prefix}_m"] = label_m
        print(label_t, label_n, label_m)
    return result_list

if __name__ == "__main__":
    input_dic = get_csv("../../../dataset/test/sample_submission.csv")
    result = process_tnm(input_dic, "gpt-3.5-turbo")
    df_result = pd.DataFrame(result)
    df_result.to_excel("test_out_3_5_lc_tnm_jp3_ignore_dual.xlsx")
