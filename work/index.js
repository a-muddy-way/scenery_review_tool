window.addEventListener("DOMContentLoaded", init);
window.addEventListener("DOMContentLoaded", init2);

function init() {


    const width = 1280;
    const height = 960;

    // レンダラーを作成
    const canvasElement = document.querySelector('#myCanvas');
    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        canvas: canvasElement,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);

    // シーンを作成
    const scene = new THREE.Scene();

    // 背景色の設定
    scene.background = new THREE.Color('#ffffff');

    let gridHelper = new THREE.GridHelper(100, 100, 0xffff00, 0x0000ff);
    scene.add(gridHelper);

    // カメラを作成
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    camera.position.set(50, 40, -50);
    camera.lookAt(0, 20, 10);

    // カメラコントローラーを作成
    const controls = new THREE.OrbitControls(camera, canvasElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.2;

    // 環境光源を作成
    const ambientLight = new THREE.AmbientLight(0xffffff);
    ambientLight.intensity = 1.0;
    scene.add(ambientLight);

    // 半球光源を作成
    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x0000FF, 1.0);
    scene.add(hemisphereLight);

    // 平行光源を作成
    const directionalLight = new THREE.DirectionalLight(0xffffff);
    directionalLight.intensity = 1.0;
    directionalLight.position.set(1, 3, 1);
    scene.add(directionalLight);

    // スポットライトを作成
    const spotLight = new THREE.SpotLight(0xFFFFFF, 1.0, 30, Math.PI / 4, 10, 0.5);
    scene.add(spotLight);

    // 3Dモデルの読み込み
    var mtlLoader = new THREE.MTLLoader();
    mtlLoader.load("output/building_base.mtl", function(materials)
    {
        materials.preload();
        var objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load("output/building_base.obj", function(object)
        {    
            plant_cube = object;
            plant_cube.position.set(-10, 0, 5);
            scene.add( plant_cube );
        });
    });

    // 画像テクスチャの読み込み
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load('input/ground.png');

    // 平面ジオメトリの作成
    const geometry = new THREE.PlaneGeometry(100, 100);
    const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = Math.PI / 2;
    scene.add(plane);

    scene.add(gridHelper);


    tick();

    function tick() {
        renderer.render(scene, camera); // rendering
        requestAnimationFrame(tick);
    }
}





function init2() {
    const width = 1280;
    const height = 960;
    const tmp_x = 60;
    const tmp_y = 0;
    const tmp_z = 20;

    // レンダラーを作成
    const canvasElement = document.querySelector('#myCanvas2');
    const renderer = new THREE.WebGLRenderer({
        antialias: true,
        canvas: canvasElement,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);

    // シーンを作成
    const scene = new THREE.Scene();

    // 背景色の設定(緑)
    scene.background = new THREE.Color('#ffffff');

    let gridHelper = new THREE.GridHelper(700, 70, 0xffff00, 0x0000ff);
    scene.add(gridHelper);

    // カメラを作成
    const camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    camera.position.set(500, 400, 400);
    camera.lookAt(20, 20, 10);

    // カメラコントローラーを作成
    const controls = new THREE.OrbitControls(camera, canvasElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.2;

    // 環境光源を作成
    const ambientLight = new THREE.AmbientLight(0xffffff);
    ambientLight.intensity = 1.0;
    scene.add(ambientLight);

    // 半球光源を作成
    const hemisphereLight = new THREE.HemisphereLight(0xffffff, 0x0000FF, 1.0);
    scene.add(hemisphereLight);

    // 平行光源を作成
    const directionalLight = new THREE.DirectionalLight(0xffffff);
    directionalLight.intensity = 1.0;
    directionalLight.position.set(1, 3, 1);
    scene.add(directionalLight);

    // スポットライトを作成
    const spotLight = new THREE.SpotLight(0xFFFFFF, 1.0, 30, Math.PI / 4, 10, 0.5);
    scene.add(spotLight);

    // 3Dモデルの読み込み
    // var fbxloader = new THREE.FBXLoader();
    // // const fbxloader = new FBXLoader();
    // fbxloader.load('output/52354611_bldg_6697_op.fbx', function (fbx) {
    //     const model = fbx.scene;
    //     // model.scale.set(0.1, 0.1, 0.1);
    //     scene.add(model);
    // });
    // 3Dモデルの読み込み
    var mtlLoader = new THREE.MTLLoader();
    mtlLoader.load("output/52354611_bldg_6697_op_LOD2.mtl", function(materials)
    {
        materials.preload();
        var objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load("output/52354611_bldg_6697_op_LOD2.obj", function(object)
        {    
            plant_cube = object;
            plant_cube.position.set(tmp_x, -40, tmp_z);
            scene.add( plant_cube );
        });
    });
    // var mtlLoader = new THREE.MTLLoader();
    mtlLoader.load("output/52354611_tran_6697_op_LOD1.mtl", function(materials)
    {
        materials.preload();
        var objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load("output/52354611_tran_6697_op_LOD1.obj", function(object)
        {    
            plant_cube = object;
            plant_cube.position.set(tmp_x, 0.1, tmp_z);
            scene.add( plant_cube );
        });
    });
    // 3Dモデルの読み込み
    var mtlLoader = new THREE.MTLLoader();
    mtlLoader.load("output/building_base.mtl", function(materials)
    {
        materials.preload();
        var objLoader = new THREE.OBJLoader();
        objLoader.setMaterials(materials);
        objLoader.load("output/building_base.obj", function(object)
        {    
            plant_cube = object;
            plant_cube.position.set(-10, 0, 5);
            scene.add( plant_cube );
        });
    });


    // 画像テクスチャの読み込み
    const textureLoader = new THREE.TextureLoader();
    const texture = textureLoader.load('input/ground.png');

    // 平面ジオメトリの作成
    const geometry = new THREE.PlaneGeometry(700, 700);
    const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(geometry, material);
    plane.rotation.x = Math.PI / 2;
    scene.add(plane);

    scene.add(gridHelper);
    const axesHelper = new THREE.AxesHelper( 100 );
    scene.add( axesHelper );


    tick();

    function tick() {
        renderer.render(scene, camera); // レンダリング
        requestAnimationFrame(tick);
    }
}