dict_synonym = {"number_of_floors": "階数", \
                "width": "幅(m)", \
                "depth": "奥行(m)", \
                "base_color_ground": "壁の色(一階)", \
                "base_color_wall": "壁の色(二階以上)", \
                "vertex_coordinates": "平面図", \
                }
lst_file_path = ['./input/pdf/guideline_takasa.pdf', './input/pdf/guideline_zentai20230406.pdf']
model_name_gpt35 = 'gpt-3.5-turbo'
model_name_gpt4o = 'gpt-4o-2024-05-13'
path_args = './output/json/args.json'
path_audio = './output/audio/response.mp3'
path_image_dir = './input/image_gpt/'
path_image_name = 'groundplan.png'
path_default_file = './output/json/default.json'
path_rag_dir = './output/chromadb'
path_tmp_file = './output/json/tmp.json'
# path_save_file = './output/json/last_saved.json'
