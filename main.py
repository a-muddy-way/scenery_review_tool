# streamlit run main.py
import asyncio  # 非同期処理を行うためのライブラリ
import base64
from dotenv import load_dotenv
import edge_tts
import json
import langchain
# /home/adminuser/venv/lib/python3.11/site-packages/langchain_core/_api/deprecation.py:119: LangChainDeprecationWarning: The function `initialize_agent` was deprecated in LangChain 0.1.0 and will be removed in 0.3.0. Use Use new agent constructor methods like create_react_agent, create_json_agent, create_structured_chat_agent, etc. instead.
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
import pandas as pd
import params as prm
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
# from tools import build_db_for_rag # 必要時のみ有効化する
from tools import get_response_by_rag, get_groundplan_function
from tools import UpdateBuildingParameterTool
from tools import remove_building_parameters, load_default_parameters, export_3D_building_model, remove_object
# from tools import load_saved_parameters
from tools import recognize_speech

# Set config
load_dotenv('.env', verbose=True)
langchain.debug = True

def init_page():
    st.set_page_config(
        page_title="XXX 景観検討ツール（モックアップ版）",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.header("XXX 景観検討ツール（モックアップ版）")
    # st.sidebar.title("Options")
    HIDE_ST_STYLE = """
        <style>
        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        div[data-testid="stDecoration"] {
            visibility: hidden;
            height: 0%;
            position: fixed;
        }
        #MainMenu {
                visibility: hidden;
                height: 0%;
        }
        header {
            visibility: hidden;
            height: 0%;
        }
        footer {
            visibility: hidden;
            height: 0%;
        }
        .appview-container .main .block-container{
            padding-top: 1rem;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 1rem;
        }  
        .reportview-container {
            padding-top: 0rem;
            padding-right: 3rem;
            padding-left: 3rem;
            padding-bottom: 0rem;
        }
        header[data-testid="stHeader"] {
                z-index: -1;
        }
        div[data-testid="stToolbar"] {
            z-index: 100;
        }
        div[data-testid="stDecoration"] {
            z-index: 100;
        }
        </style>
    """
    st.markdown(HIDE_ST_STYLE, unsafe_allow_html=True)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def get_agent(memory=None):

    with open(prm.path_default_file) as f:
        dict = json.load(f)
    lst_tmp = [k[:-2] for k in dict.keys() if k.endswith('_r') or k.endswith('_g') or k.endswith('_b')] \
         + [k for k in dict.keys() if not (k.endswith('_r') or k.endswith('_g') or k.endswith('_b'))]
    
    
    llm = ChatOpenAI(model_name=prm.model_name_gpt35, streaming=True, temperature=0)

    agent_kwargs = {
        "system_message": SystemMessage(
            content='あなたは、ユーザが建物のデザイン作成を支援するフレンドリーなアシスタントです。\
                あなたは、ユーザの入力に応じて建物のパラメータを整理します。 \
                内容が不明な項目がある場合は、ユーザーに問い合わせてください。 \
                なお、あなたは以下の機能を使用できます。 ' \
                # + '・`build_db_for_rag`： XXXの建物に関する法規をDBに登録し、検索しできるようにします。 ' \
                + '・`get_response_by_rag`： XXXの建物に関する法規を検索し、建築のアドバイスを行います。 ' \
                # + '・`get_groundplan_function`： 平面図を読み込んで建物の平面の形状を取得します。 ' \
                + '・`update_building_parameters`： ユーザが入力したパラメータを保存します。 ' \
                + '・`remove_building_parameters`： ユーザが入力したパラメータを除去します。 ' \
                + '・`load_default_parameters`： デフォルトのパラメータを取得します。 ' \
                # + '・`load_saved_parameters`： 最後に使用したパラメータを取得します。 ' \
                + '・`export_3D_building_model`： パラメータを使用して、建物の3Dモデルをエクスポートします。 ' \
                + '・`remove_object`： 3Dモデルを除去します。 ' \
                + ''
        ),
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    agent = initialize_agent(
        tools=[             
                # Tool(
                #     name = 'build_db_for_rag',
                #     func = build_db_for_rag,
                #     description = 'XXXの建物に関する法規をDBに登録し、検索しできるようにします。',
                # ),
                Tool(
                    name = 'get_response_by_rag',
                    func = get_response_by_rag,
                    description = 'ユーザからの問いに応じてXXXの建物に関する法規を検索し、建築のアドバイスを行います。',
                ),
                # Tool(
                #     name = 'get_groundplan_function',
                #     func = get_groundplan_function,
                #     description = "平面図を読みこんで、その頂点座標を取得する",
                # ),
                UpdateBuildingParameterTool(), 
                Tool(
                    name = 'remove_building_parameters',
                    func = remove_building_parameters,
                    description = 'Use this tool to remove a building parameter.' \
                        + ' Available arguments are one in the following list ' \
                        + str(list(set(lst_tmp))),
                ),
                Tool(
                    name = 'load_default_parameters',
                    func = load_default_parameters,
                    description = 'Use this tool to load the default parameters.',
                ),
                # Tool(
                #     name = 'load_saved_parameters',
                #     func = load_saved_parameters,
                #     description = 'Use this tool to load the last saved parameters.',
                # ),
                Tool(
                    name = 'export_3D_building_model',
                    func = export_3D_building_model,
                    description = 'Use this tool to export a 3D model of the building.',
                ),
                Tool(
                    name = 'remove_object',
                    func = remove_object,
                    description = 'Use this tool to remove a 3D model from the screen.',
                ),
               ],
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
        verbose=True,
    )
    return agent

def get_memory():
    # get memory from session state if it exists
    memory = st.session_state.get("memory", None)
    # initialize memory if it doesn't exist
    if memory is None:
        memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    return memory

def update_dataframe(df):
    bcg_rgb, bcw_rgb = [0, 0, 0], [0, 0, 0]
    with open(prm.path_tmp_file) as f:
        dict = json.load(f)
    for k, v in dict.items():
        if k == 'base_shape':
            pass
        elif k == 'base_color_ground_r':
            bcg_rgb[0] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcg_rgb)
            df.loc['base_color_ground'] = ['base_color_ground', prm.dict_synonym['base_color_ground'], hex_code]
        elif k == 'base_color_ground_g':
            bcg_rgb[1] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcg_rgb)
            df.loc['base_color_ground'] = ['base_color_ground', prm.dict_synonym['base_color_ground'], hex_code]
        elif k == 'base_color_ground_b':
            bcg_rgb[2] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcg_rgb)
            df.loc['base_color_ground'] = ['base_color_ground', prm.dict_synonym['base_color_ground'], hex_code]
        elif k == 'base_color_wall_r':
            bcw_rgb[0] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcw_rgb)
            df.loc['base_color_wall'] = ['base_color_wall', prm.dict_synonym['base_color_wall'], hex_code]
        elif k == 'base_color_wall_g':
            bcw_rgb[1] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcw_rgb)
            df.loc['base_color_wall'] = ['base_color_wall', prm.dict_synonym['base_color_wall'], hex_code]
        elif k == 'base_color_wall_b':
            bcw_rgb[2] = v
            hex_code = "#{:02x}{:02x}{:02x}".format(*bcw_rgb)
            df.loc['base_color_wall'] = ['base_color_wall', prm.dict_synonym['base_color_wall'], hex_code]
        elif k == 'base_shape':
            pass
        elif k == 'vertex_coordinates':
            df.loc[k] = [k, prm.dict_synonym[k], '読込済']
        elif k in prm.dict_synonym:
            df.loc[k] = [k, prm.dict_synonym[k], str(v)]
        else :
            df.loc[k] = [k, k, str(v)]
    # 表示
    return df[["df_synonym", "df_value"]].style.map(lambda x: 'background-color: ' + x if (type(x) is str and x.startswith('#')) else '')

if __name__ == "__main__":
    init_page()

    # カラム分割
    col1, col2 = st.columns(2)

    # チャット入力欄を作成
    prompt_text = st.chat_input()

    # 設定値を表示する
    st.sidebar.markdown("## 現在のデザイン条件")
    df = pd.DataFrame({'df_key': [], 'df_synonym': [], 'df_value': []})
    styler = update_dataframe(df)
    placeholder = st.sidebar.dataframe(
        styler,
        column_config = {"df_synonym": "項目", "df_value": "設定値"},
        hide_index=True,
        )

    # ファイルアップロード機能
    #  ref: https://discuss.streamlit.io/t/are-there-any-ways-to-clear-file-uploader-values-without-using-streamlit-form/40903
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    st.sidebar.markdown("### ファイルアップロード")
    with open(prm.path_args) as f:
        dict_args = json.load(f)
    file = st.sidebar.file_uploader('平面図をアップロードしてください.', 
                                    type=['jpg', 'jpeg', 'png'], 
                                    accept_multiple_files=False,
                                    key=st.session_state["file_uploader_key"],
                                    )
    if file:
        st.session_state["uploaded_files"] = file
        # st.sidebar.markdown(f'{file.name} をアップロードしました.')
        dict_args['image_path'] = file.name
        # 画像を保存する
        with open(prm.path_args, 'w') as f:
            json.dump(dict_args, f, indent=4)
        # 底面図読み込み機能を実行する
        str_out  = get_groundplan_function()
        # # 保存した画像を表示
        # st.sidebar.image(Image.open(prm.path_image_dir + dict_args['image_path']), width=400)
        # メッセージを表示
        st.session_state.messages.append({"role": "assistant", "content": str_out})
        col1.chat_message("user").write(str_out)
        st.session_state["file_uploader_key"] += 1
        st.rerun()

    # オプション
    st.sidebar.markdown("## オプション")
    st.sidebar.markdown(":red[※オンラインモックアップでは使用不可]")
    # 音声認識を実行するボタンを追加
    if st.sidebar.button('🎤 音声入力'):
        prompt_voice = recognize_speech()  # 音声認識関数を呼び出し、結果を取得
    else:
        prompt_voice = ''
    
    on_voice = st.sidebar.toggle("📢 音声出力")
    on_debug = st.sidebar.toggle("💻 開発者用")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": 'こんにちは！私は建物生成メーカーです。  \n ' \
                + '私はあなたが建物のデザインを行うために、次の機能を提供します。  \n  \n ' \

                + '  \n **:orange[＜設計の前提に関わる機能＞]**  \n ' \
                + ':orange[ ― XXXの建物に関する法規を検索し、アドバイスを行います。]  \n ' \

                + '  \n **:orange[＜デザインの作成/変更に関わる機能＞]**  \n ' \
                + ':orange[ ― デザインのひな型を取得します。  \n ' \
                + ' └ 標準では最後に使用したデザイン条件が復元されますが、はじめから作成することも可能です。]  \n '
                + ':orange[ ― 建築の平面図を読み込んで、建物の形状を取得します。  \n ' \
                + ' └ サイドバーから画像をアップロードしてください。]  \n ' \
                + ':orange[ ― 建築物のデザインを変更します。  \n ' \
                + ' └ 現在は、建物の階数・幅・奥行・壁の色を指定可能です。（1度に複数を指定可能）  \n ' \
                + ' └ 不要なデザインを削除することも可能です。（1度に1つまで指定可能）]  \n ' \

                + '  \n **:orange[＜3Dモデルの表示に関わる機能＞]**  \n ' \
                + ':orange[ ― パラメータを使用して、建物の3Dモデルをエクスポートします。]  \n ' \
                + ':orange[ ― 画面から3Dモデルを除去します。]  \n ' \

                + '  \n 前回の最後のデザイン条件を復元しました。  \n それではデザインの作成を始めましょう。'
            }
        ]
        
    for msg in st.session_state.messages:
        col1.chat_message(msg["role"]).write(msg["content"])

    if prompt_text is not None or len(prompt_voice)>0: # 入力が空ではない場合 
        prompt = ''
        if prompt_text is not None:
            prompt += prompt_text
        if len(prompt_voice):
            prompt += prompt_voice
        st.session_state.messages.append({"role": "user", "content": prompt})
        col1.chat_message("user").write(prompt)

        memory = get_memory()
        agent = get_agent(memory=memory)

        with col1.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.empty())
            response = agent.invoke(prompt, callbacks=[st_cb])
            # response = st.session_state.agent.run(prompt, callbacks=[st_cb])

            # save agent memory to session state
            st.session_state.memory = agent.memory
            st.session_state.messages.append({"role": "assistant", "content": response['output']})
            if on_debug:
                st.write(response)
            st.write(response['output'])

            # 音声を出力する
            if on_voice:
                # 音声合成の設定を行い、ファイルに保存
                communicate = edge_tts.Communicate(response, 'ja-JP-NanamiNeural', rate='+0%', pitch='+0Hz')
                asyncio.run(communicate.save(prm.path_audio))
                # 生成された音声をウェブアプリ上で再生
                # st.audio(prm.path_audio)
                autoplay_audio(prm.path_audio)

        st.rerun()
        # placeholder.dataframe(
        #     update_dataframe(pd.DataFrame({'df_key': [], 'df_synonym': [], 'df_value': []})),
        #     column_config = {"df_synonym": "項目", "df_value": "設定値"},
        #     hide_index=True,
        #     )

    col2.markdown('開発中　ー　ここに建築物の3Dモデルを表示する')
    with col2:
        html_string = '''
        <style>
            *
            {
                margin: 0;
                padding: 0;
            }

            html,
            body
            {
                overflow: hidden;
                min-height: 700px;
            }

            .webgl
            {
                position: fixed;
                top: 0;
                left: 0;
                outline: none;
            }
        </style>

        <script src="//cdnjs.cloudflare.com/ajax/libs/gsap/3.9.1/gsap.min.js"></script>
        <!--<script src="//cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.js"></script> -->
        <!-- <script src="http://threejs.org/examples/js/controls/TrackballControls.js"></script> -->


        <script type="module">
            // import * as THREE from 'https://cdn.skypack.dev/three@0.128.0/build/three.module.js';
            // import { OrbitControls } from 'https://unpkg.com/three@0.147.0/examples/js/controls/OrbitControls.js'
            // import { OBJLoader } from 'https://unpkg.com/three@0.147.0/examples/js/loaders/OBJLoader.js'
            // import { MTLLoader } from 'https://unpkg.com/three@0.147.0/examples/js/loaders/MTLLoader.js' 
            // import { fflate } from 'https://unpkg.com/three@0.147.0/examples/js/libs/fflate.min.js'
            // import * as THREE        from 'https://unpkg.com/three@0.147.0/build/three.module.js';
            
            import * as THREE        from 'https://cdn.skypack.dev/three@0.128.0/build/three.module.js';
            import { OrbitControls } from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/controls/OrbitControls.js';
            import { OBJLoader }     from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/loaders/OBJLoader.js';
            import { MTLLoader }     from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/loaders/MTLLoader.js'; 
            // import { fflate }        from 'https://cdn.skypack.dev/three@0.128.0/examples/jsm/libs/fflate.min.js';
            


            // Base
            // ----------
            
            // Initialize scene
            const scene = new THREE.Scene()
            
            // Initialize camera
            const camera = new THREE.PerspectiveCamera(30, window.innerWidth / window.innerHeight, 0.1, 60)
            
            // Reposition camera
            camera.position.set(6, 0, 0)
            
            // Initialize renderer
            const renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true
            })
            
            // Set renderer size
            renderer.setSize(window.innerWidth, window.innerHeight)
            
            // Append renderer to body
            document.body.appendChild(renderer.domElement)
            
            // Initialize controls
            const controls = new OrbitControls(camera, renderer.domElement)
            
            // World
            // ----------
            
            // Load world texture
            const worldTexture = new THREE.TextureLoader().load("https://assets.codepen.io/141041/small-world.jpg")
            
            // Initialize world geometry
            const worldGeometry = new THREE.SphereGeometry(1, 40, 40)
            
            // Initialize world material
            const worldMaterial = new THREE.MeshLambertMaterial({
            map: worldTexture
            })
            
            // Initialize world
            const world = new THREE.Mesh(worldGeometry, worldMaterial)
            
            // Add earth to scene
            scene.add(world)
            
            // Clouds
            // ----------
            
            // Load clouds texture
            const cloudTexture = new THREE.TextureLoader().load("https://assets.codepen.io/141041/small-world-clouds.png")
            
            // Initialize clouds geometry
            const cloudGeometry = new THREE.SphereGeometry(1.01, 40, 40)
            
            // Initialize clouds material
            const cloudMaterial = new THREE.MeshBasicMaterial({
            map: cloudTexture,
            transparent: true
            })
            
            // Initialize clouds
            const clouds = new THREE.Mesh(cloudGeometry, cloudMaterial)
            
            // Add clouds to scene
            scene.add(clouds)

            // add subtle ambient lighting
            const ambientLight = new THREE.AmbientLight(0xbbbbbb);
            scene.add(ambientLight);

            // directional lighting
            const directionalLight = new THREE.DirectionalLight(0xffffff);
            directionalLight.position.set(1, 1, 1).normalize();
            scene.add(directionalLight);
            
            // Animation
            // ----------      
            
            // Prepare animation loop
            function animate() {
            // Request animation frame
            requestAnimationFrame(animate)
            
            // Rotate world
            world.rotation.y += 0.0005
            
            // Rotate clouds
            clouds.rotation.y -= 0.001
            
            // Render scene
            renderer.render(scene, camera)

            }
            
            // Animate
            animate()
            
            // Resize
            // ----------
            
            // Listen for window resizing
            window.addEventListener('resize', () => {
            // Update camera aspect
            camera.aspect = window.innerWidth / window.innerHeight
            
            // Update camera projection matrix
            camera.updateProjectionMatrix()
            
            // Resize renderer
            renderer.setSize(window.innerWidth, window.innerHeight)

            });
        </script>

        <style>
        body{
            background: radial-gradient(circle at center, white, rgba(113,129,191,0.5) 50%);
        }
        </style>
        '''

        # components.html(html_string)
        # with open("index.html", "r",encoding="utf-8") as html_file:
        #     html_code = html_file.read()
        # html_code = '<body> helloworld </body> '
        # with open("index.js", "r",encoding="utf-8") as js_file:
        #     js_code = js_file.read()
        components.html(html_string, height=640)
        # col2.markdown(html_code, unsafe_allow_html=True) 
        # col2.markdown(html_string, unsafe_allow_html=True) 
