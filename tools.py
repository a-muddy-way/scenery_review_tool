# 参考；https://qiita.com/bear_montblanc/items/6ea5bb7e7e72303a8a97
import base64
import datetime
import json
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import params as prm
import pickle
from pydantic import BaseModel, Field
import re
import subprocess
import shutil
from typing import Type, Optional, List, Tuple
import speech_recognition as sr

## RAGで使用するDBを構築する
def build_db_for_rag(self):
    # PDFを読み込む
    pages = []
    for file_path in prm.lst_file_path:
        pages += PyPDFLoader(file_path).load()
    # ドキュメントをチャンクに分割
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    # ベクトルストアにドキュメントを格納
    db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory='./output/chromadb')
    return 'done'

## RAGを実行する
# def get_response_by_rag(self, str_input):
def get_response_by_rag(str_input: str):
    db = Chroma(persist_directory=prm.path_rag_dir, embedding_function=OpenAIEmbeddings())
    chat = ChatOpenAI(model_name=prm.model_name_gpt35)
    qa_chain =  RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=db.as_retriever())
    result = qa_chain.invoke(str_input)
    return result

# 平面図から座標を取得する。
# def get_groundplan_function(self):
def get_groundplan_function():
    with open(prm.path_args) as f:
        dict_args = json.load(f)

    # Getting the base64 string
    with open(prm.path_image_dir + dict_args['image_path'], "rb") as f:
        base64_image =  base64.b64encode(f.read()).decode('utf-8')
    # Call OpenAI API
    llm = ChatOpenAI(model=prm.model_name_gpt4o, temperature=0)
    res = llm.invoke(
        [
            HumanMessage(
                content = [
                    {
                        "type": "text",
                        "text": "You are an architectural design professional. \
                            You are asked to enumerate the coordinates of the vertices of the bottom figure of a building to create CAD data from the image presented. \
                            Using the bottom left vertex of the figure as the origin, indicate the coordinates of each vertex in a counterclockwise direction. \
                            Please indicate the coordinates of each of the building's bottom surfaces according to the following Json Schema." \
                            + json.dumps(BuildingGroundPlanSchema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ]
            )
        ]
    )
    with open('./output/pkl/res.pkl', 'wb') as f:
        pickle.dump(res, f)

    # データを保存する
    df = pd.DataFrame({'X': [], 'Y': []})
    for line in res.content.splitlines():
        regex = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
        lst_vertex = re.findall(rf'({regex})', line)
        
        if len(lst_vertex) >= 2:
            tpl_x = lst_vertex[len(lst_vertex)-2]
            x = float(tpl_x[len(tpl_x)-2])
            tpl_y = lst_vertex[len(lst_vertex)-1]
            y = float(tpl_y[len(tpl_y)-2])
            if(len(df[(df.X==x)&(df.Y==y)]))==0:
                df.loc[len(df)] = [x, y]

    ####
    # FixMe!!
    # ここにBlenderに底面の線分の交差や記述順序（半時計回りになっているか）のバリデーション処理を行う
    ###
    with open('./output/pkl/df.pkl', 'wb') as f:
        pickle.dump(df, f)

    lst_vertex_coordinates = []
    for row in df.itertuples():
        lst_vertex_coordinates += [(row.X, row.Y)]

    with open(prm.path_tmp_file) as f:
        dict = json.load(f)
    dict['vertex_coordinates'] = lst_vertex_coordinates
    with open(prm.path_tmp_file, 'w') as f:
        json.dump(dict, f, indent=4)

    return '平面図の読み込みが完了しました。'

class GroundVertexSchema(BaseModel):
    x_axis: float = Field(..., title="x-coordinate", description="x-coordinate of a vertex on the ground floor; plus means east and minus means west, 1.0 point means 1.0 meter")
    y_axis: float = Field(..., title="y-coordinate", description="y-coordinate of a vertex on the ground floor; plus means north and minus means south, 1.0 point means 1.0 meter")

class BuildingGroundPlanSchema(BaseModel):
    vertexs: list[GroundVertexSchema] = Field(..., title="vertexs", description="a list of vertexs on the ground floor; 1.0 point means 1.0 meter, \
                                              positive x_axis points east, negative x_axis points west, \
                                              positive y_axis points north, negative y_axis points south, ", min_length=3)

class BuildingParameterSchema(BaseModel):
    number_of_floors:    Optional[int] = Field(  3, description="number of floors of the building", ge=-1)
    width:               Optional[int] = Field( 10, description="width of the building",   ge=-1)
    depth:               Optional[int] = Field( 10, description="depth of the building",   ge=-1)
    base_color_ground_r: Optional[int] = Field(255, description="ground color (red)",   ge=-1, le=255)
    base_color_ground_g: Optional[int] = Field(127, description="ground color (green)", ge=-1, le=255)
    base_color_ground_b: Optional[int] = Field( 50, description="ground color (blue)",  ge=-1, le=255)
    base_color_wall_r:   Optional[int] = Field( 90, description="wall color (red)",     ge=-1, le=255)
    base_color_wall_g:   Optional[int] = Field( 90, description="wall color (green)",   ge=-1, le=255)
    base_color_wall_b:   Optional[int] = Field( 90, description="wall color (blue)",    ge=-1, le=255) #デフォルト値30
    # base_shape:          Optional[int] = Field(  0, description="shape of the building (0: square, 1:L )", ge=-1, le=1)
    # vertex_coordinates:  Optional[List[Tuple[float, float]]] = Field([], description='vertex coordinates of the building on a ground plan.')

class UpdateBuildingParameterTool(BaseTool):
    name = "update_building_parameters"
    description = "Use this tool to update building parameters and export 3D model."
    args_schema: Type[BuildingParameterSchema] = BuildingParameterSchema

    def _run(
        self,
        number_of_floors:   Optional[int] = -1,
        width:              Optional[int] = -1,
        depth:              Optional[int] = -1,
        base_color_ground_r: Optional[int] = -1,
        base_color_ground_g: Optional[int] = -1,
        base_color_ground_b: Optional[int] = -1,
        base_color_wall_r: Optional[int] = -1,
        base_color_wall_g: Optional[int] = -1,
        base_color_wall_b: Optional[int] = -1,
        # base_shape:        Optional[int] = -1,
        # vertex_coordinates:  Optional[List[Tuple[float, float]]] = [],
    ) -> str:
        ## FixMe!!渡された引数からパラメータを更新する。もっと行数を減らせるハズ。
        with open(prm.path_tmp_file) as f:
            dict_tmp = json.load(f)
        dict = {
            "number_of_floors": number_of_floors,
            "width": width,
            "depth": depth,
            "base_color_ground_r": base_color_ground_r,
            "base_color_ground_g": base_color_ground_g,
            "base_color_ground_b": base_color_ground_b,
            "base_color_wall_r":   base_color_wall_r,
            "base_color_wall_g":   base_color_wall_g,
            "base_color_wall_b":   base_color_wall_b,
            }
        for k in dict.keys():
            if (type(dict[k]) is int) and (dict[k] > -1):
                dict_tmp[k] = dict[k]
        with open(prm.path_tmp_file, 'w') as f:
            json.dump(dict_tmp, f, indent=4)
        return 'parameters saved: ' + json.dumps(dict_tmp)

    # 使用してしていないが宣言が必要な処理
    async def _arun(
        self,
    ):
        raise NotImplementedError("Not implemented yet")

def remove_building_parameters(str_input: str):
    str_input = str(str_input) # おまじない
    print(str_input)
    if str_input.endswith('_r') or str_input.endswith('_g') or str_input.endswith('_b') :
        # 後ろ2文字を除去する
        str_input = str_input[:-2]
    with open(prm.path_tmp_file) as f:
        dict_in = json.load(f)
    # 引数と前方一致でキーが一致しないものだけをキーとした新しい辞書を作成する。
    dict_out = {k: v for k, v in dict_in.items() if not k.startswith(str_input)}
    with open(prm.path_tmp_file, 'w') as f:
        json.dump(dict_out, f, indent=4)
    return 'Target parameter removerd: (' + str_input + ')' + json.dumps(dict_out)

def load_default_parameters(self):
    with open(prm.path_default_file) as f:
        dict = json.load(f)
    return 'default parameters loaded: ' + json.dumps(dict)

# def load_saved_parameters(self):
#     with open(prm.path_save_file) as f:
#         dict = json.load(f)
#     with open(prm.path_tmp_file, 'w') as f:
#         json.dump(dict, f, indent=4)
#     return 'saved parameters loaded: ' + json.dumps(dict)

def export_3D_building_model(self):
    cmd = 'blender --background --python mybuildify_33.py'
    cp = subprocess.run(cmd, shell=True)
    tmp = cp.returncode
    if cp.returncode == 0:
        return '処理が正常終了し、3Dモデルを出力しました' 
    else : 
        return 'エラーが発生しました。管理者に連絡してください'

def remove_object(self):
    # new_path = shutil.copy('../output/building_base.obj', '../output/building_base_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.obj')
    # new_path = shutil.copy('../output/dummy.obj', '../output/building_base.obj')
    shutil.copy('../output/building_base.obj', '../output/building_base_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.obj')
    shutil.copy('../output/dummy.obj', '../output/building_base.obj')
    return 'ファイルのリセットが完了しました。'

# 音声認識を行う関数の定義
def recognize_speech():
    r = sr.Recognizer()  # 音声認識器のインスタンスを作成
    mic = sr.Microphone()  # マイクロフォンのインスタンスを作成
    with mic as source:  # マイクロフォンを音声入力ソースとして使用
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)  # 音声を聞いて、オーディオデータを取得
        try:
            # Googleの音声認識サービスを使用して日本語のテキストに変換
            return r.recognize_google(audio, language="ja-JP")
        except sr.UnknownValueError:
            # 音声が認識できなかった場合
            return ''
        except sr.RequestError:
            # 音声認識サービスへのリクエストに失敗した場合
            return ''