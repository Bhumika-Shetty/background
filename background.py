import io
import os
import PySimpleGUI as sg
import shutil
import tempfile
from PIL import Image, ImageColor, ImageDraw
import base64
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import torch
import skimage.filters
from torchvision.transforms.functional import to_tensor, to_pil_image

def load_model():
    return torch.jit.load('./torchscript_resnet50_fp32.pth')

def get_segmentation_shadow(model, original_image, background_image):
    global shadow_intensity, shadow_blur
    x, y = 0, 0

    # print(original_image.size)

    if original_image.size[0]%4:
        x = original_image.size[0] - original_image.size[0]%4
    
    if original_image.size[1]%4:
        y = original_image.size[1] - original_image.size[1]%4

    if x or y:
        if x == 0:
            x = original_image.size[0]
        if y == 0:
            y = original_image.size[1]

        original_image = original_image.crop((0,0,x,y))
        # original_image.show()

    bgr = original_image.crop((0,0,100, 100)).resize(original_image.size)
    src_tensor = torch.unsqueeze(to_tensor(original_image), 0)
    bgr_tensor = torch.unsqueeze(to_tensor(bgr), 0)

    if src_tensor.size(2) <= 2048 and src_tensor.size(3) <= 2048:
        model.backbone_scale = 1/4
        model.refine_sample_pixels = 80_000
    else:
        model.backbone_scale = 1/8
        model.refine_sample_pixels = 320_000

    alpha_, fgr_ = model(src_tensor, bgr_tensor)[:2]

    shadow_ = (1-alpha_)[0]
    shadow_[0][shadow_[0] < shadow_intensity] = shadow_intensity 

    bg_img = background_image.resize(original_image.size)
    bg_tensor = torch.unsqueeze(to_tensor(bg_img), 0)

    blur_bk = skimage.filters.gaussian(shadow_, sigma=shadow_blur)
    shadow_bg = torch.tensor(blur_bk)*bg_tensor


    
    com = (1-alpha_) * shadow_bg + alpha_ * fgr_
    return to_pil_image(com[0]), to_pil_image(shadow_bg[0])

def get_segmentation(model, original_image, background_image):
  bgr = original_image.crop((0,0,100, 100)).resize(original_image.size)
  src_tensor = torch.unsqueeze(to_tensor(original_image), 0)
  bgr_tensor = torch.unsqueeze(to_tensor(bgr), 0)
  bg_img = background_image.resize(original_image.size)  
  bg_tensor = torch.unsqueeze(to_tensor(bg_img), 0)

  if src_tensor.size(2) <= 2048 and src_tensor.size(3) <= 2048:
    model.backbone_scale = 1/4
    model.refine_sample_pixels = 80_000
  else:
    model.backbone_scale = 1/8
    model.refine_sample_pixels = 320_000

  alpha_, fgr_ = model(src_tensor, bgr_tensor)[:2]
  com = alpha_ * fgr_ + (1 - alpha_) * bg_tensor  
  return to_pil_image(com[0])

def load_segmented_image(filename):
    global im, im_np, im_h, im_w, im_64, segmented_img, segmented_img_np, graph, image, color, original_img, original_img_np, scale, background_image, background_image_np
    # print("original_img size = ", original_img.size)
    scale = 1
    # try:
    if os.path.exists(filename):
        background_image = Image.open(filename)
        img, background_image = get_segmentation_shadow(model, original_img, background_image)
        background_image_np = np.array(background_image)

        with BytesIO() as f:
            img.save(f, format='PNG')
            f.seek(0)
            im = Image.open(f)
            im.load()

            segmented_img = Image.open(f)
            segmented_img.load()

            im_64 = base64.b64encode(f.getvalue())
        im_np = np.array(im)
        im_w, im_h = im.size
        image = graph.DrawImage(data = im_64, location=(0, 0) )
        segmented_img_np = np.array(segmented_img)  
        color = pick_color((0,0))
        sg.popup("make sure you also uploaded the original image")
    else:
        sg.popup("some error has occur please upload the image again")
    # except:
    #     sg.popup('original image is not loaded please click the load original image')

def load_original_image(filename):
    global original_img, original_img_np
    if os.path.exists(filename):
        original_img = Image.open(filename)
        with BytesIO() as f:
            original_img.save(f, format='PNG')
            f.seek(0)
            original_img = Image.open(f)
            original_img.load()
        original_img_np = np.array(original_img)
        sg.popup(f"uploaded: {filename}")
    else:
        print("error while loading the original image")

def do_zoom(scale, resample):
    global image, window, im_np, original_img_np, original_img, segmented_img, segmented_img_np, background_image, background_image_np
    graph.delete_figure(image)
    # current_scale = scale/current_scale


    im = Image.fromarray(im_np)
    im = im.resize((int(im_w*scale), int(im_h*scale)),resample=resample)
    im_np = np.array(im)

    original_img = original_img.resize ((int(im_w*scale), int(im_h*scale)),resample=resample)
    original_img_np = np.array(original_img)

    segmented_img = segmented_img.resize ((int(im_w*scale), int(im_h*scale)),resample=resample)
    segmented_img_np = np.array(segmented_img)

    background_image = background_image.resize ((int(im_w*scale), int(im_h*scale)),resample=resample)
    background_image_np = np.array(background_image)

    with io.BytesIO() as buffer:
        im.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    image = graph.DrawImage(data = im_64, location=(0, 0) )

    graph.Widget.configure(width=im_w*scale, height=im_h*scale)
    max_width = max(im_w*scale, width)
    max_height = max(im_h*scale, height)
    canvas = window['Column1'].Widget.canvas
    canvas.configure(scrollregion=(0, 0, max_width, max_height))

    # print(im)

def zoom_in():
    global scale
    if scale < 4:
        scale += 1
    # if scale < 4 and scale >= 1:
    #     scale += 1
    # elif scale >= 1/4 and scale < 1:
    #     scale += scale
    # print(scale)
    resample = Image.NEAREST
    do_zoom(scale, resample)

def zoom_out():
    global scale
    if scale > 1:
        scale -= 1
    # elif scale <= 1 and scale > 1/4:
    #     scale /=2
    # print(scale)
    resample = Image.LANCZOS
    do_zoom(scale, resample)

def convert_hex_color(color):
    d = {'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15}
    # print(color)
    try: 
        r1 = int(color[1])
    except:
        r1 = d[color[1]]
    
    try: 
        r2 = int(color[2])
    except:
        r2 = d[color[2]]
    
    try: 
        g1 = int(color[3])
    except:
        g1 = d[color[3]]

    try: 
        g2 = int(color[4])
    except:
        g2 = d[color[4]]

    try: 
        b1 = int(color[5])
    except:
        b1 = d[color[5]]
        
    try: 
        b2 = int(color[6])
    except:
        b2 = d[color[6]]

    return [r1 * 16 + r2, g1*16 + g2, b1*16 + b2]

def draw_poly(poly_vertices):
    global current_polygon
    if current_polygon:
        graph.delete_figure(current_polygon)
        # print("deleted polygon = ",current_polygon)
    current_polygon = graph.draw_polygon(poly_vertices, fill_color = '', line_color = 'black', line_width = 1)
    # print("new polygon = ", current_polygon)
    # print(poly_vertices)

def replace_selection(poly_vertices):

    if len(poly_vertices) <1:
        return []

    x_list = []
    y_list_ = []
    for point in poly_vertices:
        x_list.append(point[0])
        y_list_.append(point[1])

    x_max = max(x_list)
    y_max = max(y_list_)
    x_min = min(x_list)
    y_min = min(y_list_)
    
    ext = 1
    y_list = y_list_
    x, y = np.meshgrid(np.arange(x_min - ext, x_max + ext), np.arange(y_min - ext, y_max + ext)) 
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    verts = [(x,y) for x,y in zip(x_list, y_list)]
    path = Path(verts)
    grid = path.contains_points(points, radius=-1e-9)

    true_list_x = x[grid]
    true_list_y = y[grid]

    false_list_x = x[grid==False]
    false_list_y = y[grid == False]

    to_replace_point_list = [(x,y) for x,y in zip(true_list_x, true_list_y)]
    
    for point in poly_vertices:
        to_replace_point_list.append(point)
    return to_replace_point_list

def replace_with_original(poly_vertices):
    #read the original image
    # original_filename = '/Users/amanankesh/Desktop/image_editor/PySimpleGUI/image.png'
    # original_img = Image.open(original_filename)
    # original_img_np = np.array(original_img)
    global original_img_np
    #get the replacement points
    replacement_points = replace_selection(poly_vertices)
    #make changes 
    for point in replacement_points:
        im_np[point[1]][point[0]] = original_img_np[point[1]][point[0]]

    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace_selection_pen(poly_vertices):

    if len(poly_vertices) <1:
        return []

    x_list = []
    y_list_ = []
    for point in poly_vertices:
        x_list.append(point[0])
        y_list_.append(point[1])

    x_max = max(x_list)
    y_max = max(y_list_)
    x_min = min(x_list)
    y_min = min(y_list_)
    
    ext = 1
    y_list = y_list_
    x, y = np.meshgrid(np.arange(x_min - ext, x_max + ext), np.arange(y_min - ext, y_max + ext)) 
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    
    verts = [(x,y) for x,y in zip(x_list, y_list)]
    path = Path(verts)
    grid = path.contains_points(points, radius=0.9)

    true_list_x = x[grid]
    true_list_y = y[grid]

    false_list_x = x[grid==False]
    false_list_y = y[grid == False]

    to_replace_point_list = [(x,y) for x,y in zip(true_list_x, true_list_y)]
    
    for point in poly_vertices:
        to_replace_point_list.append(point)
    return to_replace_point_list

def change_using_pen(points):
    global im_np, scale, im_h, im_w, size
    bg_color = convert_hex_color(color)
    replacement_points = replace_selection_pen(points)
    s = int(size/2)
    if s == 0:
        s = 1
    for point in replacement_points:
        w = point[1]
        h = point[0]
        for x in range(w-s, w+s):
            for y in range(h-s, h+s):
                if x > -1 and x< scale*im_h and y > -1 and y < scale*im_w:
                    im_np[x][y] =  bg_color

    # print("points = ", points)
    # print("replacement_points = ", replacement_points)

def replace_with_color_poly_select(poly_vertices):
    #read the original image
    global im_np
    bg_color = convert_hex_color(color)
    #get the replacement points
    replacement_points = replace_selection(poly_vertices)
    #make changes 
    for point in replacement_points:
        im_np[point[1]][point[0]] = bg_color

    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace_with_segmented_image_poly_select(poly_vertices):
    #read the original image
    # segmented_img = Image.open(filename)
    # segmented_img_np = np.array(segmented_img)
    global segmented_img_np
    #get the replacement points
    replacement_points = replace_selection(poly_vertices)
    #make changes v          
    for point in replacement_points:
        im_np[point[1]][point[0]] = segmented_img_np[point[1]][point[0]]

    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace_with_background_image_poly_select(poly_vertices):
    #read the original image
    # segmented_img = Image.open(filename)
    # segmented_img_np = np.array(segmented_img)
    global background_image_np
    #get the replacement points
    replacement_points = replace_selection(poly_vertices)
    #make changes 
    for point in replacement_points:
        im_np[point[1]][point[0]] = background_image_np[point[1]][point[0]]

    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def pick_color(point):
    color = im_np[point[1]][point[0]]

    r = str(hex(color[0]))[2:]
    if len(r) < 2:
        r = '0' + r

    g = str(hex(color[1]))[2:]
    if len(g) < 2:
        g = '0' + g

    b = str(hex(color[2]))[2:]
    if len(b) < 2:
        b = '0' + b

    return '#' + r + g + b

def undo():
    # print(figure_list)
    l = len(figure_list)
    if l:
        idlist = figure_list[l - 1]
        for i in range(len(idlist)-1, -1, -1):
            graph.delete_figure(idlist[i])
            idlist.pop()
        figure_list.pop()

def replace_with_original_image(im, x1, y1, x2, y2):
    #read the original image
    # original_filename = '/Users/amanankesh/Desktop/image_editor/PySimpleGUI/image.png'
    # original_img = Image.open(original_filename)
    # original_img_np = np.array(original_img)
    global original_img_np
    # im_np = np.array(im)   
    #make changes 
    min_y = min(x1, x2)
    min_x = min(y1, y2)
    max_y = max(x1, x2)+1
    max_x = max(y1, y2)+1
    im_np[min_x:max_x, min_y: max_y, : ] = original_img_np[min_x:max_x, min_y: max_y, : ]
    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace(im, x1, y1, x2, y2):
    # im_np = np.array(im)   
    global im_np
    bg_color = convert_hex_color(color)
    #make changes 
    min_y = min(x1, x2)
    min_x = min(y1, y2)
    max_y = max(x1, x2)+1
    max_x = max(y1, y2)+1
    im_np[min_x:max_x, min_y: max_y, : ] = bg_color
    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace_with_segmented_image(im, x1, y1, x2, y2):
    #read the original image
    global segmented_img_np
    # im_np = np.array(im)   
    #make changes 
    min_y = min(x1, x2)
    min_x = min(y1, y2)
    max_y = max(x1, x2)+1
    max_x = max(y1, y2)+1
    im_np[min_x:max_x, min_y: max_y, : ] = segmented_img_np[min_x:max_x, min_y: max_y, : ]
    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def replace_with_background_image(im, x1, y1, x2, y2):
    #read the original image
    global background_image_np
    # im_np = np.array(im)   
    #make changes 
    min_y = min(x1, x2)
    min_x = min(y1, y2)
    max_y = max(x1, x2)+1
    max_x = max(y1, y2)+1
    im_np[min_x:max_x, min_y: max_y, : ] = background_image_np[min_x:max_x, min_y: max_y, : ]
    #convert to pil
    im_pil = Image.fromarray(im_np)
    with io.BytesIO() as buffer:
        im_pil.save(buffer, format="PNG")
        data = buffer.getvalue()
    im_64 = base64.b64encode(data)
    return im_64

def save_image(values):
    global im_np
    save_filename = sg.popup_get_file(
        "File", file_types=file_types, save_as=True, no_window=True
    )
    if save_filename == values["segmented_file"]:
        sg.popup_error(
            "You are not allowed to overwrite the original image!")
    else:
        if save_filename:
            pil_img = Image.fromarray(im_np)
            pil_img.save(save_filename)
            sg.popup(f"Saved: {save_filename}")

file_types = [("JPEG (*.jpg)", "*.jpg"), ("All files (*.*)", "*.*")]
im = ''
im_np = ''
im_w, im_h = 0,0
width, height = 1600, 1200 # 1500, 1300 
im_64 = ''


segmented_img = ''
segmented_img_np = ''

background_img = ''
background_img = ''

original_img = ''
original_img_np = ''
key = '-graph-'

image = 0

scale = 1
current_scale = 1

image_column= [
    [
        sg.Graph(canvas_size=(width, height), graph_bottom_left=(0,height), graph_top_right=(width, 0), background_color='white',drag_submits = True, enable_events = True, key=key)
    ], 
]
button_column = [ 
    [   

        sg.Text("Original Image File"),
        sg.Input(
            size=(25, 1), key="original_file"
        ),
        sg.FileBrowse(file_types=file_types),
        sg.Button("Load Original Image"),

        sg.Text("background"),
        sg.Input(
            size=(25, 1), key="segmented_file"
        ),
        sg.FileBrowse(file_types=file_types),
        sg.Button("Load Segmented Image"),
    ],    
    [
        sg.Text("shadow intensity"), 
        sg.Slider(range = (0, 100), default_value = 50, tick_interval = 10, orientation= 'horizontal', enable_events = True, size=(50,20), key='shadow_intensity'), 
        sg.Text("shadow blur"), 
        sg.Slider(range = (0, 100), default_value = 50, tick_interval = 10, orientation= 'horizontal', enable_events = True, size=(50,20), key='shadow_blur'), 

    ],
    [   
        sg.Button('Replace'), 
        sg.Button('Replace with original image'), 
        sg.Button('Replace with segmented image'), 
        sg.Button('Replace with background image'), 
        sg.Button('undo'), sg.Button('select'), 
        sg.Button('pen'), sg.Text("pen size"), 
        sg.Slider(range = (1, 5), default_value = 1, tick_interval = 1, orientation= 'horizontal', enable_events = True, key='pen_size'), 
        sg.Button('pick color'), 
        sg.Button ('poly select'), sg.Button('+'), 
        sg.Button('- '), sg.Button('show graph'),
        sg.Button('Save')
    ]
]
layout = [    
        [sg.Column(image_column,  scrollable=True, key = "Column1")],

        [button_column],   
]

window = sg.Window('Graph', layout=layout, resizable=True, finalize=True)
graph = window[key]
image = graph.DrawImage(data = im_64, location=(0, 0) )

down_flag = False
start_postion = (0,0)
end_postion = (0,0)


rectanlge_id_list = []
point_list = []
figure_list = []
id_list = []
poly_vertices = []

current_rectangle = 0
# pointColor = 'black'
# lineColor = 'blue'

# variables to enable different tools
pen = False
select = False
color_pick = False


#for polygon select 
size = 1
color = 0
current_polygon = 0
poly_select = False
poly_vertices = []

model = load_model()
shadow_intensity = 0.5
shadow_blur = 5

while True:      
    event, values = window.read() 
    # print("events = ", event)
    # print("values = ", values)

    if event == sg.WIN_CLOSED:      
        break   

    elif event == key:
        if select:
            for id in rectanlge_id_list:
                if id:
                    graph.delete_figure(id)

            if (down_flag == False):
                start_postion = values[key]
                down_flag = True
            end_postion = values[key]
            current_rectangle = graph.DrawRectangle(start_postion, end_postion, line_color='black')
            rectanlge_id_list.append(current_rectangle)
        
        elif pen:
            point_list.append(values[key])
            start_point = values[key]
            for i in range(len(point_list) - 1):
                line_id = graph.DrawLine(point_list[i], point_list[i+1], color, size)
                change_using_pen([point_list[i], point_list[i+1], point_list[i]])
                # print(line_id)
                id_list.append(line_id)
                del point_list[i]
                
            point_id = graph.DrawPoint(start_point, size, color)
            # im_np[start_point[1]][start_point[0]] = convert_hex_color(color)
            id_list.append(point_id)
        
    elif event == key + '+UP':
        if select:
            end_postion = values[key]
            down_flag = False

        elif pen:
            point_list = []
            figure_list.append(id_list)
            id_list = []

        elif color_pick == True:
            color = pick_color(values[key])
            # print(color)

        elif poly_select:
            poly_vertices.append(values[key])
            if len(poly_vertices) > 0:
                draw_poly(poly_vertices)

    elif event == 'Replace with original image':
        if select:
            graph.delete_figure(image)
            img = replace_with_original_image(im, start_postion[0], start_postion[1], end_postion[0], end_postion[1])
            image = graph.DrawImage(data = img, location = (0, 0))

        elif poly_select:
            graph.delete_figure(image)
            img = replace_with_original(poly_vertices)
            image = graph.DrawImage(data = img, location = (0, 0))
            poly_vertices = []

    elif event == 'Replace':
        if select:
            graph.delete_figure(image)
            img = replace(im, start_postion[0], start_postion[1], end_postion[0], end_postion[1])
            image = graph.DrawImage(data = img, location = (0, 0))
        
        if poly_select:
            graph.delete_figure(image)
            img = replace_with_color_poly_select(poly_vertices)
            image = graph.DrawImage(data = img, location = (0, 0))
            poly_vertices = []

    elif event == 'Replace with segmented image':
        if select:
            graph.delete_figure(image)
            img = replace_with_segmented_image(im, start_postion[0], start_postion[1], end_postion[0], end_postion[1])
            image = graph.DrawImage(data = img, location = (0, 0))
        if poly_select:
            graph.delete_figure(image)
            img = replace_with_segmented_image_poly_select(poly_vertices)
            image = graph.DrawImage(data = img, location = (0, 0))
            poly_vertices = []

    elif event == 'Replace with background image':
        if select:
            graph.delete_figure(image)
            img = replace_with_background_image(im, start_postion[0], start_postion[1], end_postion[0], end_postion[1])
            image = graph.DrawImage(data = img, location = (0, 0))
        if poly_select:
            graph.delete_figure(image)
            img = replace_with_background_image_poly_select(poly_vertices)
            image = graph.DrawImage(data = img, location = (0, 0))
            poly_vertices = []

    elif event == 'pen':
        select = False
        color_pick = False
        poly_select = False
        pen = True
        if current_rectangle:
            graph.delete_figure(current_rectangle)
            rectanlge_id_list.pop()
        
    elif event == 'select':
        pen = False
        color_pick = False
        poly_select = False
        select = True
    
    elif event == 'undo':
        undo()

    elif event == 'pick color':
        if current_rectangle:
            graph.delete_figure(current_rectangle)
            rectanlge_id_list.pop()
        pen = False
        select = False
        poly_select = False        
        color_pick = True

    elif event == 'poly select':
        pen = False
        color_pick = False
        select = False
        poly_select = True
        # print("current polygon = ", current_polygon)
        if current_polygon:
            graph.delete_figure(current_polygon)
        poly_vertices = []
    
    elif event == '+':
        zoom_in()
    elif event == '- ':
        zoom_out()
    elif event =='show graph':
        img = Image.fromarray(im_np)
        img.save('tmp.png')
        # img.show()

    elif event == 'Load Segmented Image':
        name = values['segmented_file']
        load_segmented_image(name)

    elif event == 'Load Original Image':
        name = values['original_file']
        load_original_image(name)

    elif event == "Save" and values["segmented_file"]:
        save_image(values)

    elif event == 'pen_size':
        size = values['pen_size']

    elif event == 'shadow_intensity':
        shadow_intensity = values['shadow_intensity']/10 # controls the darkness

    elif event == 'shadow_blur':
        shadow_blur = values['shadow_blur'] #sigma for guassian blur

window.close()