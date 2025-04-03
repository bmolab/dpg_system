# note sometimes it is necessary to reinstall cairo
# conda install -c conda-forge pycairo
# after conda uninstall pycairo first



import cairo
import math
import numpy as np
import platform

_initialized = False
import ctypes as ct
colour_rate = 256


def set_up_canvas(frame):
    height = frame[3] - frame[1]
    width = frame[2] - frame[0]
    image = np.zeros(shape=(height, width, 3))
    alpha = np.zeros(shape=(height, width, 1))
    alpha.fill(255)
    cat_list = (image, alpha)
    dest_image = np.concatenate([d.astype(np.uint8) for d in cat_list], axis=2)
    stride = cairo.ImageSurface.format_stride_for_width(cairo.FORMAT_RGB24, dest_image.shape[1])
    surface = cairo.ImageSurface.create_for_data(dest_image, cairo.FORMAT_RGB24, dest_image.shape[1],
                                                 dest_image.shape[0], stride)
    cr = cairo.Context(surface)
    cr.set_line_width(2)
    cr.set_source_rgb(1.0, 1.0, 1.0)
    cr.set_operator(cairo.OPERATOR_LIGHTEN)
    return cr, dest_image


def draw__(cr, text):
    cr.set_source_rgb(1.0, 1.0, 1.0)
 #   cr.move_to()
    cr.stroke()


class PycairoContext(ct.Structure):
    _fields_ = \
        [
            ("PyObject_HEAD", ct.c_byte * object.__basicsize__),
            ("ctx", ct.c_void_p),
            ("base", ct.c_void_p),
        ]


# end PycairoContext
def create_cairo_font_face_for_file (filename, faceindex=0, loadoptions=0):
    "given the name of a font file, and optional faceindex to pass to FT_New_Face" \
    " and loadoptions to pass to cairo_ft_font_face_create_for_ft_face, creates" \
    " a cairo.FontFace object that may be used to render text with that font."
    global _initialized
    global _freetype_so
    global _cairo_so
    global _ft_lib
    global _ft_destroy_key
    global _surface

    CAIRO_STATUS_SUCCESS = 0
    FT_Err_Ok = 0

    if not _initialized:
        # find shared objects
        if platform.system() == "Darwin":
            _freetype_so = ct.CDLL("libfreetype.6.dylib")
            _cairo_so = ct.CDLL("libcairo.2.dylib")
        elif platform.system() == "Linux":
            _freetype_so = ct.CDLL("libfreetype.so.6")
            _cairo_so = ct.CDLL("libcairo.so.2")

        _cairo_so.cairo_ft_font_face_create_for_ft_face.restype = ct.c_void_p
        _cairo_so.cairo_ft_font_face_create_for_ft_face.argtypes = [ ct.c_void_p, ct.c_int ]
        _cairo_so.cairo_font_face_get_user_data.restype = ct.c_void_p
        _cairo_so.cairo_font_face_get_user_data.argtypes = (ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_font_face_set_user_data.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p)
        _cairo_so.cairo_set_font_face.argtypes = [ ct.c_void_p, ct.c_void_p ]
        _cairo_so.cairo_font_face_status.argtypes = [ ct.c_void_p ]
        _cairo_so.cairo_font_face_destroy.argtypes = (ct.c_void_p,)
        _cairo_so.cairo_status.argtypes = [ ct.c_void_p ]
        # initialize freetype
        _ft_lib = ct.c_void_p()
        status = _freetype_so.FT_Init_FreeType(ct.byref(_ft_lib))
        if  status != FT_Err_Ok :
            raise RuntimeError("Error %d initializing FreeType library." % status)
        #end if

        # class PycairoContext(ct.Structure):
        #     _fields_ = \
        #         [
        #             ("PyObject_HEAD", ct.c_byte * object.__basicsize__),
        #             ("ctx", ct.c_void_p),
        #             ("base", ct.c_void_p),
        #         ]
        # #end PycairoContext

        _surface = cairo.ImageSurface(cairo.FORMAT_A8, 0, 0)
        _ft_destroy_key = ct.c_int() # dummy address
        _initialized = True
    #end if

    ft_face = ct.c_void_p()
    cr_face = None
    try :
        # load FreeType face
        status = _freetype_so.FT_New_Face(_ft_lib, filename.encode("utf-8"), faceindex, ct.byref(ft_face))
        if status != FT_Err_Ok :
            raise RuntimeError("Error %d creating FreeType font face for %s" % (status, filename))
        #end if

        # create Cairo font face for freetype face
        cr_face = _cairo_so.cairo_ft_font_face_create_for_ft_face(ft_face, loadoptions)
        status = _cairo_so.cairo_font_face_status(cr_face)
        if status != CAIRO_STATUS_SUCCESS :
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        #end if
        # Problem: Cairo doesn't know to call FT_Done_Face when its font_face object is
        # destroyed, so we have to do that for it, by attaching a cleanup callback to
        # the font_face. This only needs to be done once for each font face, while
        # cairo_ft_font_face_create_for_ft_face will return the same font_face if called
        # twice with the same FT Face.
        # The following check for whether the cleanup has been attached or not is
        # actually unnecessary in our situation, because each call to FT_New_Face
        # will return a new FT Face, but we include it here to show how to handle the
        # general case.
        if _cairo_so.cairo_font_face_get_user_data(cr_face, ct.byref(_ft_destroy_key)) == None :
            status = _cairo_so.cairo_font_face_set_user_data \
              (
                cr_face,
                ct.byref(_ft_destroy_key),
                ft_face,
                _freetype_so.FT_Done_Face
              )
            if status != CAIRO_STATUS_SUCCESS :
                raise RuntimeError("Error %d doing user_data dance for %s" % (status, filename))
            #end if
            ft_face = None # Cairo has stolen my reference
        #end if

        # set Cairo font face into Cairo context
        cairo_ctx = cairo.Context(_surface)
        cairo_t = PycairoContext.from_address(id(cairo_ctx)).ctx
        _cairo_so.cairo_set_font_face(cairo_t, cr_face)
        status = _cairo_so.cairo_font_face_status(cairo_t)
        if status != CAIRO_STATUS_SUCCESS :
            raise RuntimeError("Error %d creating cairo font face for %s" % (status, filename))
        #end if

    finally :
        _cairo_so.cairo_font_face_destroy(cr_face)
        _freetype_so.FT_Done_Face(ft_face)
    #end try

    # get back Cairo font face as a Python object
    face = cairo_ctx.get_font_face()
    return face
#end create_cairo_font_face_for_file


# face = create_cairo_font_face_for_file("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0)
# surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 128, 128)
# ctx = cairo.Context(surface)
#
# ctx.set_font_face(face)
# ctx.set_font_size(30)
# ctx.move_to(0, 44)
# ctx.show_text("Hello,")
# ctx.move_to(30, 74)
# ctx.show_text("world!")
#
# del ctx
#
# surface.write_to_png("hello.png")

