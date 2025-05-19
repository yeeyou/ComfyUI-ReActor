import os
import shutil
# Ensure these are imported at the top of reactor_swapper.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading # Optional for logging thread ID
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

import insightface
from insightface.app.common import Face
# try:
#     import torch.cuda as cuda
# except:
#     cuda = None
import torch

import folder_paths
import comfy.model_management as model_management
from modules.shared import state

from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
)
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]
# if cuda is not None:
#     if cuda.is_available():
#         providers = ["CUDAExecutionProvider"]
#     else:
#         providers = ["CPUExecutionProvider"]
# else:
#     providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)

# NUM_PARALLEL_TASKS_HARDCODED = 10 # Hardcode for testing
FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640)):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # if order == "large-small":
    #     return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
        face,
        face_index,
        gender_condition,
        operated: str,
        order: str,
):
    gender = [
        x.sex
        for x in face
    ]
    gender.reverse()
    # If index is outside of bounds, return None, avoid exception
    if face_index >= len(gender):
        logger.status("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(gender))
        return None, 0
    face_gender = gender[face_index]
    logger.status("%s Face %s: Detected Gender -%s-", operated, face_index, face_gender)
    if (gender_condition == 1 and face_gender == "F") or (gender_condition == 2 and face_gender == "M"):
        logger.status("OK - Detected Gender matches Condition")
        try:
            faces_sorted = sort_by_order(face, order)
            return faces_sorted[face_index], 0
            # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
        except IndexError:
            return None, 0
    else:
        logger.status("WRONG - Detected Gender doesn't match Condition")
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 1
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 1

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = getAnalysisModel(det_size)
    faces = face_analyser.get(img_data)

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            logger.status(f"Swapping...")
                            if face_boost_enabled:
                                logger.status(f"Face Boost is enabled")
                                bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                M *= scale
                                result = swapper.in_swap(target_img, bgr_fake, M)
                            else:
                                # logger.status(f"Swapping as-is")
                                result = face_swapper.get(result, target_face, source_face)
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            # Keep searching for other faces if wrong gender is detected, enhancement
                            #if source_face_idx == len(source_faces_index):
                            #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                            #    return result_image
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.status(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        # Keep searching for other faces if wrong gender is detected, enhancement
                        #if source_face_idx == len(source_faces_index):
                        #    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        #    return result_image
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_image

# NEW WORKER FUNCTION (to be placed before swap_face_many)
def _swap_face_many_worker(
    target_img_cv2_bgr: np.ndarray,
    target_faces_analyzed: list, # List of Face objects for this specific image
    source_face_obj: Face,       # The single selected source Face object
    faces_index_to_swap: list[int],
    gender_target_setting: int,
    faces_order_setting: list[str],
    face_swapper_instance,      # The loaded FaceSwap model instance (e.g., from getFaceSwapModel)
    face_boost_enabled_flag: bool,
    face_restore_model_for_boost,
    face_restore_visibility_for_boost,
    codeformer_weight_for_boost,
    interpolation_for_boost,
    original_image_index: int # For debugging and result ordering
) -> tuple[int, np.ndarray]: # Returns original index and processed CV2 BGR image

    thread_id = threading.get_ident() # Optional: for logging
    logger.debug(f"[Worker {thread_id}] Processing image index {original_image_index}")
    
    if not target_faces_analyzed: # No faces detected on this target image
        logger.debug(f"[Worker {thread_id}] No faces in target image index {original_image_index}, returning original.")
        return original_image_index, target_img_cv2_bgr

    # Important: Work on a copy if multiple faces_index entries might modify the same image buffer
    # sequentially within this worker. If faces_index always has one element, or if
    # face_swapper.get always returns a new image, a copy at the start is safest.
    current_target_cv2_bgr = target_img_cv2_bgr.copy()

    for face_num in faces_index_to_swap:
        if face_num >= len(target_faces_analyzed):
            logger.debug(f"[Worker {thread_id}] Target face index {face_num} out of bounds for image {original_image_index}. Skipping.")
            continue

        # Note: get_face_single might need the original target_img for analysis if it re-analyzes.
        # Here, we pass current_target_cv2_bgr, assuming get_face_single works on pre-analyzed faces_list.
        # The `target_img` argument to `get_face_single` in your original code was `target_img` (the cv2 one for the current loop).
        # And `target_face` was `target_faces` (the list of Face objects for that image).
        # So, it should be:
        target_face_single, wrong_gender = get_face_single(
            current_target_cv2_bgr, # Image data for context if get_face_single needs it (e.g. if it re-analyzes, though it shouldn't here)
            target_faces_analyzed,  # The list of Face objects for this image
            face_index=face_num,
            gender_target=gender_target_setting, # gender_target_setting is the correct name
            order=faces_order_setting[0] # target face order
        )

        if target_face_single is not None and wrong_gender == 0:
            logger.debug(f"[Worker {thread_id}] Swapping face {face_num} in image {original_image_index}...")
            if face_boost_enabled_flag:
                logger.debug(f"[Worker {thread_id}] Face Boost enabled for image {original_image_index}.")
                try:
                    # Ensure face_swapper_instance is the correct object (from getFaceSwapModel)
                    bgr_fake, M = face_swapper_instance.get(current_target_cv2_bgr, target_face_single, source_face_obj, paste_back=False)
                    
                    # Face Boost restoration (scripts.r_faceboost.restorer and .swapper)
                    # This part needs to be thread-safe if restorer/swapper have shared state
                    # or are not re-entrant. If they are simple functions or stateless classes, it's fine.
                    # For now, assume they are safe to call like this.
                    bgr_fake, scale = restorer.get_restored_face(
                        bgr_fake, 
                        face_restore_model_for_boost, 
                        face_restore_visibility_for_boost, 
                        codeformer_weight_for_boost, 
                        interpolation_for_boost
                    )
                    M *= scale
                    current_target_cv2_bgr = swapper.in_swap(current_target_cv2_bgr, bgr_fake, M) # Pass the current state of the image
                except Exception as e_boost:
                    logger.error(f"[Worker {thread_id}] Error during Face Boost for image {original_image_index}: {e_boost}")
                    # Fallback to simple swap if boost fails
                    current_target_cv2_bgr = face_swapper_instance.get(current_target_cv2_bgr, target_face_single, source_face_obj, paste_back=True)
            else:
                current_target_cv2_bgr = face_swapper_instance.get(current_target_cv2_bgr, target_face_single, source_face_obj, paste_back=True)
        elif wrong_gender == 1:
            logger.debug(f"[Worker {thread_id}] Wrong target gender for face {face_num} in image {original_image_index}. Skipping.")
        else:
            logger.debug(f"[Worker {thread_id}] No target face for index {face_num} in image {original_image_index}. Skipping.")
            
    return original_image_index, current_target_cv2_bgr

# --- NEW WORKER FUNCTION FOR PARALLEL FACE ANALYSIS ---
def _analyze_single_target_image_worker(
    pil_img: Image.Image, 
    original_image_index: int
) -> tuple[int, np.ndarray, list]: # Returns: original_idx, cv2_bgr_array, list_of_faces
    thread_id = threading.get_ident()
    # 使用 logger.status 以便在你的环境中更容易看到输出
    logger.status(f"[AnalyzeWorker {thread_id} HACK] Analyzing image original_index: {original_image_index}")
    try:
        cv2_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # 确保 analyze_faces 返回的是 Face 对象列表或空列表
        faces_found = analyze_faces(cv2_bgr) 
        logger.status(f"[AnalyzeWorker {thread_id} HACK] Finished image original_index: {original_image_index}, found {len(faces_found if faces_found else [])} faces.")
        return original_image_index, cv2_bgr, faces_found if faces_found else []
    except Exception as e_analyze:
        logger.error(f"[AnalyzeWorker {thread_id} HACK] Error analyzing image original_index: {original_image_index}: {e_analyze}")
        try:
            # 尝试返回原始转换后的图像和空人脸列表作为回退
            cv2_bgr_fallback = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return original_image_index, cv2_bgr_fallback, []
        except: 
            # 如果连转换都失败，返回一个明确的错误指示或一个空的小图像
            logger.error(f"[AnalyzeWorker {thread_id} HACK] Critical error converting image original_index: {original_image_index} for fallback.")
            return original_image_index, np.zeros((10,10,3), dtype=np.uint8), []


def swap_face_many(
    source_img: Union[Image.Image, None],    # PIL Image from ReActor node
    target_imgs: List[Image.Image],          # List of PIL Images from ReActor node
    model: Union[str, None] = None,          # ONNX model filename
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],            # Target face indices to swap in each image
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,    # Pre-analyzed source Face object (or list)
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,              # For Face Boost
    face_restore_visibility: float = 1.0,   # Changed to float for visibility
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
    num_threads: int = 4,                # Number of threads for parallel processing
):
    # Global cache variables (their management remains mostly the same)
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH 
    # TARGET_FACES and TARGET_IMAGE_HASH were for single target, less relevant here

    logger.status(f"[swap_face_many HACK] Entered. Processing {len(target_imgs)} target images. Parallelism to be applied.")

    if not target_imgs or model is None:
        logger.warning("[swap_face_many HACK] No target images or model specified. Returning originals.")
        return target_imgs

    # --- 1. Prepare Source Face (once) ---
    actual_source_face_obj: Union[Face, None] = None
    cv2_source_img_for_analysis = None # Only if source_img is PIL and no face_model

    if face_model is not None:
        logger.status("[swap_face_many HACK] Using provided face_model as source.")
        if isinstance(face_model, list): # If a list of faces is provided for source
            if not face_model:
                logger.error("[swap_face_many HACK] Provided face_model is an empty list.")
                return target_imgs
            s_idx = source_faces_index[0] if source_faces_index and 0 <= source_faces_index[0] < len(face_model) else 0
            actual_source_face_obj = face_model[s_idx]
        else: # Assuming it's a single Face object
            actual_source_face_obj = face_model
    elif source_img is not None: # PIL Image
        logger.status("[swap_face_many HACK] Analyzing source PIL image.")
        # Convert PIL to CV2 BGR for analysis
        cv2_source_img_for_analysis = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        
        # Use MD5 hash for caching source analysis
        source_md5 = get_image_md5hash(cv2_source_img_for_analysis)
        if SOURCE_IMAGE_HASH == source_md5 and SOURCE_FACES is not None:
            logger.status("[swap_face_many HACK] Using cached source faces.")
            analyzed_s_faces = SOURCE_FACES
        else:
            logger.status("[swap_face_many HACK] Analyzing new source image for faces.")
            analyzed_s_faces = analyze_faces(cv2_source_img_for_analysis)
            SOURCE_FACES = analyzed_s_faces # Cache it
            SOURCE_IMAGE_HASH = source_md5   # Cache hash
        
        if not analyzed_s_faces:
            logger.error("[swap_face_many HACK] No faces found in source image.")
            return target_imgs
        
        # Select the specific source face from analyzed_s_faces
        # The get_face_single function is primarily for target selection with gender.
        # For source, we might need a simpler selection or adapt get_face_single.
        # Original code used: source_face, src_wrong_gender = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
        # Let's adapt this for our analyzed_s_faces
        actual_source_face_obj, src_wrong_gender = get_face_single(
            cv2_source_img_for_analysis, # image data
            analyzed_s_faces,            # list of Face objects
            face_index=source_faces_index[0] if source_faces_index else 0,
            gender_source=gender_source, # gender_source used here for selection
            gender_target=0,             # Not relevant for source selection step
            order=faces_order[1]         # Source face order
        )
        if src_wrong_gender == 1: # if gender condition not met for selected source face
             logger.warning("[swap_face_many HACK] Selected source face does not meet gender criteria. This might lead to unexpected results if gender is strict.")
             # Depending on strictness, you might want to return or try another source face.
             # For now, we proceed with the selected face (which might be None if get_face_single returns None on gender mismatch)

    else:
        logger.error("[swap_face_many HACK] No source (image or face_model) provided.")
        return target_imgs

    if actual_source_face_obj is None:
        logger.error("[swap_face_many HACK] Failed to obtain a valid source face object.")
        return target_imgs
    
    logger.status(f"[swap_face_many HACK] Successfully selected source face. Gender (0=F,1=M): {getattr(actual_source_face_obj, 'sex', 'N/A')}")


    # --- 2. Load Face Swapper Model (once) ---
    # This instance will be shared by all threads. Its internal ONNX session must be thread-safe
    # (ONNX Runtime sessions generally are) and configured with SessionOptions (Phase 1).
    if "inswapper" in model:
        model_path = os.path.join(insightface_path, model)
    elif "reswapper" in model: # Assuming reswapper models are directly in reswapper_path
        model_path = os.path.join(reswapper_path, model)
    else: # Fallback or error for unknown model type
        logger.error(f"[swap_face_many HACK] Unknown model type in name: {model}. Assuming insightface path.")
        model_path = os.path.join(insightface_path, model)

    if not os.path.exists(model_path):
        logger.error(f"[swap_face_many HACK] Swapper model file not found: {model_path}")
        return target_imgs
        
    try:
        logger.info(f"[swap_face_many HACK] Getting FaceSwapModel for: {model_path}")
        face_swapper = getFaceSwapModel(model_path) # This should use the patched __init__
    except Exception as e_load_swapper:
        logger.error(f"[swap_face_many HACK] Error loading FaceSwapModel: {e_load_swapper}")
        return target_imgs

    # --- 3. Parallel Pre-analysis of all target images for faces ---
    # --- THIS SECTION IS REPLACED ---
    logger.status(f"[swap_face_many HACK] Parallel pre-analyzing {len(target_imgs)} target images using {num_threads} threads...")
    
    analysis_results_ordered = [(None, None)] * len(target_imgs) # (cv2_bgr_img, list_of_faces_on_img)

    if not target_imgs: # 处理空 target_imgs 列表的情况
        logger.warning("[swap_face_many HACK] target_imgs list is empty before analysis.")
    elif num_threads == 1 or len(target_imgs) == 1: # 如果只有一个线程或一张图，串行分析
        logger.status(f"[swap_face_many HACK] Analyzing {len(target_imgs)} target image(s) serially (due to num_threads=1 or single image).")
        for i, pil_img_target in enumerate(target_imgs):
            _idx, cv2_bgr_res, faces_found_res = _analyze_single_target_image_worker(pil_img_target, i)
            analysis_results_ordered[i] = (cv2_bgr_res, faces_found_res)
    else: # 并行分析
        with ThreadPoolExecutor(max_workers=num_threads) as analyzer_executor:
            future_to_analysis_idx = {}
            for i, pil_img_target in enumerate(target_imgs):
                future = analyzer_executor.submit(_analyze_single_target_image_worker, pil_img_target, i)
                future_to_analysis_idx[future] = i
            
            logger.status(f"[swap_face_many HACK] All {len(future_to_analysis_idx)} analysis tasks submitted. Waiting for completion...")
            for future in as_completed(future_to_analysis_idx):
                original_idx = future_to_analysis_idx[future]
                try:
                    _idx, cv2_bgr_res, faces_found_res = future.result()
                    analysis_results_ordered[original_idx] = (cv2_bgr_res, faces_found_res)
                except Exception as e_future_analyze:
                    logger.error(f"[swap_face_many HACK] Error during parallel analysis for image index {original_idx}: {e_future_analyze}")
                    # Fallback for this specific image
                    try:
                        pil_img_fallback = target_imgs[original_idx] # Get original PIL
                        cv2_bgr_fallback_analyze = cv2.cvtColor(np.array(pil_img_fallback), cv2.COLOR_RGB2BGR)
                        analysis_results_ordered[original_idx] = (cv2_bgr_fallback_analyze, [])
                    except Exception as e_crit_fallback:
                         logger.error(f"[swap_face_many HACK] Critical fallback error for image {original_idx}: {e_crit_fallback}")
                         analysis_results_ordered[original_idx] = (np.zeros((10,10,3), dtype=np.uint8), []) # Last resort placeholder
    
    logger.status(f"[swap_face_many HACK] Finished parallel pre-analysis. Results count: {len(analysis_results_ordered)}")

    # 将分析结果转换为 targets_data_for_swap_workers 列表 (格式: [{'original_idx': i, 'cv2_img': ..., 'faces': ...}, ...])
    targets_data_for_workers = []
    for i in range(len(target_imgs)): # 使用 len(target_imgs) 来确保索引正确
        if i < len(analysis_results_ordered) and analysis_results_ordered[i][0] is not None:
            cv2_bgr_img, faces_on_img = analysis_results_ordered[i]
        else: # 如果 analysis_results_ordered 由于某种原因没有对应项，或分析失败返回了 None
            logger.warning(f"[swap_face_many HACK] Missing or failed analysis result for original image index {i}. Using placeholder.")
            # 创建一个黑色图像或从原始 target_imgs[i] 转换（如果还想尝试）
            try:
                cv2_bgr_img = cv2.cvtColor(np.array(target_imgs[i]), cv2.COLOR_RGB2BGR) if target_imgs[i] else np.zeros((100,100,3), dtype=np.uint8)
            except:
                cv2_bgr_img = np.zeros((100,100,3), dtype=np.uint8) # 应急的黑色图像
            faces_on_img = []
        targets_data_for_workers.append({'original_idx': i, 'cv2_img': cv2_bgr_img, 'faces': faces_on_img})
    # --- END OF REPLACED SECTION ---
    
    # --- 4. Parallel Processing using ThreadPoolExecutor ---
    processed_cv2_images = [None] * len(target_imgs)
    # NUM_PARALLEL_TASKS_HARDCODED = num_threads # 直接使用 num_threads

    if len(target_imgs) == 1 or num_threads == 1:
        logger.status(f"[swap_face_many HACK] Processing face swap for {len(target_imgs)} image(s) serially.") # 使用 logger.status
        # --- 修改串行循环 ---
        for item_data in targets_data_for_workers: # 迭代字典列表
            original_idx = item_data['original_idx']
            cv2_bgr_img_to_process = item_data['cv2_img']
            faces_on_img_to_process = item_data['faces']
            
            # 确保 cv2_bgr_img_to_process 是 NumPy 数组
            if not isinstance(cv2_bgr_img_to_process, np.ndarray):
                logger.error(f"[swap_face_many HACK] Serial: Item at index {original_idx} 'cv2_img' is not a NumPy array, but {type(cv2_bgr_img_to_process)}. Skipping.")
                processed_cv2_images[original_idx] = cv2_bgr_img_to_process # 或者一个错误占位符
                continue
            
            try:
                _idx, result_img = _swap_face_many_worker(
                    cv2_bgr_img_to_process, faces_on_img_to_process, actual_source_face_obj,
                    faces_index, gender_target, faces_order, face_swapper, # face_swapper 是加载的模型实例
                    face_boost_enabled, face_restore_model, face_restore_visibility,
                    codeformer_weight, interpolation, original_idx # 传递 original_idx
                )
                processed_cv2_images[original_idx] = result_img
            except Exception as e_serial_worker:
                logger.error(f"[swap_face_many HACK] Error in serial swap worker for image {original_idx}: {e_serial_worker}")
                processed_cv2_images[original_idx] = cv2_bgr_img_to_process # Fallback to original (which is already a CV2 BGR array)
    else:
        logger.status(f"[swap_face_many HACK] Processing face swap for {len(target_imgs)} image(s) with {num_threads} parallel tasks (swap stage).") # 使用 logger.status
        with ThreadPoolExecutor(max_workers=num_threads) as executor: # Renamed to 'executor' from 'swap_executor' for consistency
            future_to_idx = {}
            # --- 修改并行任务提交循环 ---
            for item_data in targets_data_for_workers: # 迭代字典列表
                original_idx_submit = item_data['original_idx']
                cv2_bgr_img_submit = item_data['cv2_img']
                faces_on_img_submit = item_data['faces']

                # 确保 cv2_bgr_img_submit 是 NumPy 数组
                if not isinstance(cv2_bgr_img_submit, np.ndarray):
                    logger.error(f"[swap_face_many HACK] Parallel Submit: Item at index {original_idx_submit} 'cv2_img' is not a NumPy array, but {type(cv2_bgr_img_submit)}. Skipping submit for this item.")
                    # 需要处理这种情况，例如直接将原始（可能是错误的）数据放入结果，或者跳过
                    # 为了简单，我们先跳过提交，后续结果收集时会用原始数据填充
                    processed_cv2_images[original_idx_submit] = cv2_bgr_img_submit # 预填充原始（可能是错误的）
                    continue

                future = executor.submit(
                    _swap_face_many_worker,
                    cv2_bgr_img_submit, faces_on_img_submit, actual_source_face_obj,
                    faces_index, gender_target, faces_order, face_swapper, # face_swapper 是加载的模型实例
                    face_boost_enabled, face_restore_model, face_restore_visibility,
                    codeformer_weight, interpolation, original_idx_submit # 传递 original_idx_submit
                )
                future_to_idx[future] = original_idx_submit # 使用 original_idx_submit
            
            logger.status(f"[swap_face_many HACK] All {len(future_to_idx)} swap tasks submitted. Waiting for completion...") # 使用 logger.status
            for future in as_completed(future_to_idx):
                idx_completed = future_to_idx[future] # 这是正确的 original_idx
                try:
                    _returned_idx, result_cv2_img = future.result()
                    processed_cv2_images[idx_completed] = result_cv2_img
                except Exception as e_future: # Renamed from e_future_swap
                    logger.error(f"[swap_face_many HACK] Exception from swap worker for image index {idx_completed}: {e_future}")
                    # Fallback to original image
                    # targets_data_for_workers[idx_completed]['cv2_img'] 应该是原始的CV2 BGR图像
                    if idx_completed < len(targets_data_for_workers) and isinstance(targets_data_for_workers[idx_completed]['cv2_img'], np.ndarray):
                        processed_cv2_images[idx_completed] = targets_data_for_workers[idx_completed]['cv2_img']
                    else: # 如果连原始数据都找不到了或类型不对，放一个黑色占位符
                        logger.error(f"[swap_face_many HACK] Fallback error: Could not retrieve original cv2_img for index {idx_completed}")
                        ph, pw = target_imgs[idx_completed].height if idx_completed < len(target_imgs) else 100, target_imgs[idx_completed].width if idx_completed < len(target_imgs) else 100
                        processed_cv2_images[idx_completed] = np.zeros((ph, pw, 3), dtype=np.uint8)

    # --- 5. Convert processed CV2 BGR images back to PIL RGB ---
    final_pil_results = []
    for cv2_img in processed_cv2_images:
        if cv2_img is not None:
            pil_rgb = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            final_pil_results.append(pil_rgb)
        else:
            # This case should ideally not happen if fallback to original is implemented
            logger.error("[swap_face_many HACK] A processed image was None. This indicates an issue.")
            pass 
            
    logger.status(f"[swap_face_many HACK] Finished. Returning {len(final_pil_results)} processed PIL images.")
    return final_pil_results