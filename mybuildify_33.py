import bpy
import bmesh
import json
import logging 
import os
import sys

class CustomException(Exception):
    def __init__(self, arg=""):
        self.arg = arg

class MyScriptException(CustomException):
    def __str__(self):
        return (
            f'[Error: {self.arg}]'
        )

def purge_orphans():
    """
    Remove all orphan data blocks

    see this from more info:
    https://youtu.be/3rNqVPtbhzc?t=149
    """
    bpy.ops.outliner.orphans_purge(num_deleted=0, do_local_ids=True, do_linked_ids=True, do_recursive=True)

def clean_scene():
    """
    Removing all of the objects, collection, materials, particles,
    textures, images, curves, meshes, actions, nodes, and worlds from the scene

    Checkout this video explanation with example

    "How to clean the scene with Python in Blender (with examples)"
    https://youtu.be/3rNqVPtbhzc
    """
    # make sure the active object is not in Edit Mode
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()

    # make sure non of the objects are hidden from the viewport, selection, or disabled
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    # select all the object and delete them (just like pressing A + X + D in the viewport)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # find all the collections and remove them
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])

    # in the case when you modify the world shader
    # delete and recreate the world object
    world_names = [world.name for world in bpy.data.worlds]
    for name in world_names:
        bpy.data.worlds.remove(bpy.data.worlds[name])
    # create a new world data block
    bpy.ops.world.new()
    bpy.context.scene.world = bpy.data.worlds["World"]

    purge_orphans()

def remove_unrequired_faces(in_bm):
    cut_num = 1
    bpy.ops.object.mode_set(mode='EDIT')
    bmesh.ops.subdivide_edges(in_bm, edges=in_bm.edges, cuts=cut_num, use_grid_fill=True)

    bpy.ops.object.mode_set(mode='OBJECT')
    faces_set = set()
    faces_delete_target = list()
    faces_x, faces_y = set(), set()

    for face in in_bm.faces:
        location_vector = (face.verts[0].co+face.verts[1].co+face.verts[2].co+face.verts[3].co)/4
        if str(location_vector.x)+'-'+str(location_vector.y)+'-'+str(location_vector.z) in faces_set:
            faces_delete_target.append(face)
        else:
            faces_set.add(str(round(location_vector.x, 3))+'-'+str(round(location_vector.y, 3))+'-'+str(round(location_vector.z, 3)))
            faces_x.add(round(location_vector.x, 3))
            faces_y.add(round(location_vector.y, 3))
    bmesh.ops.delete(in_bm, geom = faces_delete_target, context = 'FACES' )
    faces_x_sorted, faces_y_sorted = sorted(list(faces_x)), sorted(list(faces_y))
    
    faces_conv_dict = dict()
    for face in in_bm.faces:
        location_vector = (face.verts[0].co+face.verts[1].co+face.verts[2].co+face.verts[3].co)/4
        # print('face index: ' +  str(face.index) + ', face center: '  + str(location_vector))
        faces_conv_dict[faces_y_sorted.index(round(location_vector.y, 3)) *(cut_num+1) + faces_x_sorted.index(round(location_vector.x, 3))] = face.index
    print(faces_conv_dict)
    return faces_conv_dict

def remove_unrequired_faces_sub(in_bm):
    cut_num = 2
    bpy.ops.object.mode_set(mode='EDIT')
    bmesh.ops.subdivide_edges(in_bm, edges=in_bm.edges, cuts=cut_num, use_grid_fill=True)

    bpy.ops.object.mode_set(mode='OBJECT')
    faces_set = set()
    faces_delete_target = list()
    faces_x, faces_y = set(), set()

    for face in in_bm.faces:
        location_vector = (face.verts[0].co+face.verts[1].co+face.verts[2].co+face.verts[3].co)/4
        if str(location_vector.x)+'-'+str(location_vector.y)+'-'+str(location_vector.z) in faces_set:
            faces_delete_target.append(face)
        else:
            faces_set.add(str(round(location_vector.x, 3))+'-'+str(round(location_vector.y, 3))+'-'+str(round(location_vector.z, 3)))
            faces_x.add(round(location_vector.x, 3))
            faces_y.add(round(location_vector.y, 3))
    bmesh.ops.delete(in_bm, geom = faces_delete_target, context = 'FACES' )
    faces_x_sorted, faces_y_sorted = sorted(list(faces_x)), sorted(list(faces_y))
    
    faces_conv_dict = dict()
    for face in in_bm.faces:
        location_vector = (face.verts[0].co+face.verts[1].co+face.verts[2].co+face.verts[3].co)/4
        # print('face index: ' +  str(face.index) + ', face center: '  + str(location_vector))
        faces_conv_dict[faces_y_sorted.index(round(location_vector.y, 3)) *(cut_num+1) + faces_x_sorted.index(round(location_vector.x, 3))] = face.index
    print(faces_conv_dict)
    return faces_conv_dict

def main():
    ## パラメータ定義
    # リテラルを宣言する
    input_filepath_buildfy = os.path.join(os.path.dirname(__file__), './input/buildify_1.0.blend')
    output_filepath_obj = os.path.join(os.path.dirname(__file__), './output/building_base.obj')
    inner_path = 'Object'
    src_object_name = 'building_base'
    dst_object_name = 'Plane'
    
    # オブジェクトのパラメータを読み込む
    with open(os.path.join(os.path.dirname(__file__), './output/json/tmp.json')) as f:
        building_parameters = json.load(f)

    ## 不要なデータを消去する
    clean_scene()

    ## 新しいコレクションにPlaneを作成する #FIXME!! コレクションにリンクされていない 
    # コレクションを作成してシーンにリンクする
    # my_colection = bpy.data.collections.new('MyCollection')
    # bpy.context.scene.collection.children.link(my_colection)
    # アクティブなコレクションを指定する
    # layer_collection = bpy.context.view_layer.layer_collection.children[my_colection.name]
    # bpy.context.view_layer.active_layer_collection = layer_collection
    # メッシュを作成する
    my_mesh = bpy.data.meshes.new("MyMesh")
    # オブジェクトを作成する
    new_obj = bpy.data.objects.new(dst_object_name, my_mesh)
    # 現在のシーンにオブジェクトをリンクさせる
    bpy.context.scene.collection.objects.link(new_obj)
    # 作成したオブジェクトをアクティブオブジェクトにする
    bpy.context.view_layer.objects.active = new_obj
    # 作成したオブジェクトを選択状態にする
    new_obj.select_set(True)
    # BMeshを作成する
    bm = bmesh.new()
    # 面を作成する
    if 'vertex_coordinates' in building_parameters.keys():
        verts, lst_x, lst_y = [], [], [] 
        for v in building_parameters['vertex_coordinates']:
            lst_x += [v[0]]
            lst_y += [v[1]]
        for v in building_parameters['vertex_coordinates']:
            verts += [bm.verts.new([v[0] / max(lst_x) * building_parameters['depth'], \
                                        v[1] / max(lst_y) * building_parameters['width'], \
                                        0.0])]
            bm.faces.new(verts)
            bm.to_mesh(my_mesh)
    else:
        verts = [bm.verts.new([0.0, 0.0, 0.0]), \
                bm.verts.new([building_parameters['depth'], 0.0, 0.0]), \
                bm.verts.new([building_parameters['depth'], building_parameters['width'], 0.0]), \
                bm.verts.new([0.0, building_parameters['width'], 0.0]), \
                ]
        bm.faces.new(verts)
        bm.to_mesh(my_mesh)

        if 'base_shape' in building_parameters.keys():
            # 面を加工する
            bm.faces.ensure_lookup_table() # おなじない
            bm.faces[0].select = True
            bm.from_mesh(bpy.context.object.data)
            if 'base_shape' in building_parameters.keys():
                if building_parameters['base_shape'] == 1:
                    plane_index_conv_dict = remove_unrequired_faces(bm)
                    bm.faces.ensure_lookup_table() # おなじない
                    bmesh.ops.delete(bm, geom = [bm.faces[plane_index_conv_dict[2]]], context = 'FACES' )
                elif building_parameters['base_shape'] == 2:
                    plane_index_conv_dict = remove_unrequired_faces_sub(bm)
                    bm.faces.ensure_lookup_table() # おなじない
                    bmesh.ops.delete(bm, geom = [bm.faces[plane_index_conv_dict[1]]], context = 'FACES' )
            # bm.to_mesh(my_mesh)

    ## 新しいコレクションにBuildfyを読み込む
    # コレクションを作成してシーンにリンクする
    buildify_colection = bpy.data.collections.new('BuildifyCollection')
    bpy.context.scene.collection.children.link(buildify_colection)
    # アクティブなコレクションを指定する
    layer_collection = bpy.context.view_layer.layer_collection.children[buildify_colection.name]
    bpy.context.view_layer.active_layer_collection = layer_collection

    # オブジェクトをシーンに追加（Append）する
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.append(
        filepath  = os.path.join(input_filepath_buildfy, inner_path, src_object_name),
        directory = os.path.join(input_filepath_buildfy, inner_path),
        filename  = src_object_name
        )
    # Modifierのパラメータを設定してコピーする
    mysrc  = bpy.data.objects[src_object_name]
    for n in mysrc.modifiers['GeometryNodes'].node_group.nodes:
        # 不要な屋上のオブジェクトを削除する
        if n.name == 'Group.007':
            n.inputs['Props density'].default_value = 0.00
        # 不要なカンバンオブジェクトを削除する
        if n.name == 'Group.008':
            n.inputs['Props density'].default_value = 0.00

    if bpy.app.version >= (4, 0, 0):
        mysrc.modifiers['GeometryNodes'].node_group.nodes['Group Input'].interface.items_tree["Min number of floors"].default_value = building_parameters['number_of_floors'] + 1 
        mysrc.modifiers['GeometryNodes'].node_group.nodes['Group Input'].interface.items_tree["Max number of floors"].default_value = building_parameters['number_of_floors'] + 1 
    else:
        mysrc.modifiers['GeometryNodes'].node_group.inputs["Min number of floors"].default_value = building_parameters['number_of_floors'] + 1 
        mysrc.modifiers['GeometryNodes'].node_group.inputs["Max number of floors"].default_value = building_parameters['number_of_floors'] + 1 

    mydist = bpy.data.objects[dst_object_name]
    for mod in mysrc.modifiers:
        # To avoid copying a known modifier
        mod_copy = mydist.modifiers.new(mod.name, mod.type)
        for attr in sorted(dir(mod)):
            if (attr.startswith("_") or attr in ["bl_rna"]):
                continue
            if (mod.is_property_readonly(attr)):
                continue
            setattr(mod_copy, attr, getattr(mod, attr))
    # Buildfyを非表示にする
    buildify_colection.hide_viewport = True
    mydist.modifiers["GeometryNodes"].show_viewport = True

    # 1Fの壁の色を変える
    if 'base_color_ground_r' in building_parameters.keys() and \
       'base_color_ground_g' in building_parameters.keys() and \
       'base_color_ground_b' in building_parameters.keys():
        ground_material = bpy.data.materials['proxy_mat_stone']
        ground_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value \
            = ( round((building_parameters['base_color_ground_r']+100)/(255+100), 3), \
                round((building_parameters['base_color_ground_g']+100)/(255+100), 3), \
                round((building_parameters['base_color_ground_b']+100)/(255+100), 3), \
                1.0)
    # 2Fの壁の色を変える
    if 'base_color_wall_r' in building_parameters.keys() and \
       'base_color_wall_g' in building_parameters.keys() and \
       'base_color_wall_b' in building_parameters.keys():
        wall_material = bpy.data.materials['proxy_mat_yellow']
        wall_material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value \
            = ( round((building_parameters['base_color_wall_r']+100)/(255+100), 3), \
                round((building_parameters['base_color_wall_g']+100)/(255+100), 3), \
                round((building_parameters['base_color_wall_b']+100)/(255+100), 3), \
                1.0)

    ## ObjectをExportする
    # 対象のオブジェクトを選択する
    bpy.data.objects[dst_object_name].select_set(True)
    bpy.ops.wm.obj_export(filepath= output_filepath_obj)

logging.basicConfig(filename='./log/logger_mybuildify.log', level=logging.DEBUG)

# ロガーとファイルハンドラの作成
lgr = logging.getLogger('mylogger', )
lgr.setLevel(logging.DEBUG)
fhr = logging.FileHandler('./log/logger_mybuildify.log', mode='a', encoding='utf-8')
fhr.setLevel(logging.DEBUG)
fhr.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s :%(message)s'))
lgr.addHandler(fhr)
lgr.debug('Start Process: mybuildify')

main()
# try:
#     main()
#     lgr.debug('(0)  Completed: mybuildify')
#     sys.exit(0)
# except(MyScriptException) as e:
#     lgr.error('(98) Terminated by Considered Error: mybuildify \n '+ str(e), stack_info=True)
#     sys.exit(98)    
# except(Exception) as e:
#     lgr.error('(99) Terminated by Unconsidered Error: mybuildify \n '+ str(e), stack_info=True)
#     sys.exit(99)
