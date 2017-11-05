# image-to-paint

Generate a series of brushstrokes from an image. There are several parameters to vary the style and can also export GCODE for plotting.

This is a cleanup of an old school project from 2012. There are many code and methode improvments to be made.

Requires pygame and PIL

## Examples

```
 ./imageToPaint example/apple.jpg apple_out.jpg --brushes 2 4 
```


```
 ./imageToPaint example/humanoid.jpg humanoid_out.jpg --brushes 4 8

```


## Other options
./imageToPaint -h

```
usage: imageToPaint    [-h] [--brushes BRUSHES [BRUSHES ...]]
                       [--max_len MAX_LEN] [--min_len MIN_LEN]
                       [--f_grid F_GRID] [--colors COLORS [COLORS ...]]
                       [--lightdark LIGHTDARK] [--t T] [--f_l F_L] [--f_c F_C]
                       [--gcode] [--gc_feed GC_FEED] [--gc_tool GC_TOOL]
                       [--gc_width GC_WIDTH] [--gc_height GC_HEIGHT]
                       in out
	
positional arguments:
  in                    Path to input image.
  out                   Path to where to save output image.
	
optional arguments:
  -h, --help            show this help message and exit
  --brushes BRUSHES [BRUSHES ...]
                        Size of brushes strokes in pixels.
  --max_len MAX_LEN     Maximum stroke Length.
  --min_len MIN_LEN     Maximum stroke Length.
  --f_grid F_GRID       How spaced apart new strokes will be, 1 =
                        1*radius_stroke apart.
  --colors COLORS [COLORS ...]
                        Use a limited color palate.
  --lightdark LIGHTDARK
                        Paint brush strokes in increasing brightness.
  --t T                 Threshold to create a new stroke.
  --f_l F_L             Higher values will result in longer, less acurate
                        brushstrokes.
  --f_c F_C             Curvature modifier.
  --gcode
  --gc_feed GC_FEED     GCODE feedrate..
  --gc_tool GC_TOOL     GCODE tool.
  --gc_width GC_WIDTH   Width (in inches) for gcode.
  --gc_height GC_HEIGHT
                        Height (in inches) for gcode.
```
