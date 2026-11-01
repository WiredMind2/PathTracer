# A hybrid renderer in CUDA

You get a window and fly a camera through a static 3D scene. The picture is 1920 by 1080 and the target is 60 frames per second. OpenGL draws what the camera sees. CUDA shades each hit with one hard shadow and, on mirrors, one reflection. Python describes the scene. The fly loop stays in native code.

The machine this design is written for is the RTX 4050 Laptop in this box, with 6141 MiB of memory, driver 616.56, and CUDA 13.0. Those figures are from `nvidia-smi` and `nvcc`.

## The scene is a static triangle soup

World space is right-handed and Y is up. Python passes one soup. The native side copies it to the GPU once. After that, only the live camera changes.

Each vertex has a position and a normal. Each triangle has three vertex indices and one material index. A matte material stores an RGB albedo. A mirror material stores only the tag. There is one point light, with a position and an RGB intensity. Intensity does not fall off with distance. The scene also stores a starting camera: eye, target, up, and a vertical field of view in degrees.

On the GPU the resident data is the vertex buffer, the triangle list, the material table, the light, and a bounding hierarchy over the triangles. The hierarchy is a binary tree. The CPU builds it at load by splitting triangle centroids at their midpoint. The kernel only walks that tree. The build does not run again during flight.

The live camera is a different record from the starting camera. Keys and the mouse write the live camera. They do not write the scene.

## One frame rasters, then shades

The window is fixed at 1920 by 1080. It does not resize.

Each frame does four things.

1. The window reads the keys and the mouse and updates the live camera. W and S move forward and back on the horizontal plane. A and D strafe. The camera stays upright. The mouse changes yaw and pitch, and pitch stops short of looking straight up or straight down. Escape, or closing the window, ends the flight. Move speed and mouse sensitivity are constants in the window, not fields on the scene.
2. OpenGL draws the soup. Front faces are counterclockwise. Back faces are culled. A hit pixel stores world position, world normal, albedo, and the material tag. A miss stays empty.
3. One CUDA kernel reads those pixels. The geometry targets and the color texture are shared with CUDA through OpenGL interop. Every hit shoots one shadow ray at the point light. A mirror pixel also shoots one reflection ray. The kernel writes a color texture.
4. The window draws that texture.

The shadow ray and the reflection ray both start a short step along the normal, 0.001 world units, so a surface does not hit itself. Scenes in the tests are room-sized, with positions on the order of 1.

A surface is in shadow when the hierarchy returns a hit strictly between that origin and the light.

The incoming view direction points from the camera to the surface. The reflection direction is that vector reflected about the normal.

Matte color is linear. The ambient coefficient is 0.1. The clamped cosine is the greater of 0 and the dot product of the unit normal and the unit direction toward the light. A shadowed matte pixel, or a pixel whose clamped cosine is 0, is albedo times 0.1. A lit matte pixel is albedo times 0.1, plus albedo times the light intensity times the clamped cosine.

A mirror pixel does not show its own albedo. If the reflection ray hits a matte surface, the pixel uses that matte color, including that surface's own shadow. If the reflection ray hits a mirror, the kernel shades that hit with the matte formula and an albedo of 1, 1, 1, and it does not shoot a second reflection. If the reflection ray misses, or the rasterizer missed the pixel, the color is the background. The background is the constant linear RGB 0.08, 0.09, 0.11. There is no tone map. Values written for display are clamped to the range from 0 to 1. Test lights are chosen so the shade result is already inside that range.

## Python checks the scene, then the native side trusts it

The import is `engine`. The repo directory stays `PathTracer`.

Python builds a scene and calls one of two functions. `run(scene)` opens a visible window, uploads the soup, builds the hierarchy, and flies until Escape or the window closes. `render(scene, camera)` draws one frame at 1920 by 1080 through the same raster and shade path and returns the pixels. The camera argument has eye, target, up, and a vertical field of view in degrees, the same fields as the starting camera. The return value is a 1080 by 1920 by 3 array of 32-bit floats, linear RGB, with row 0 at the top of the window. `render` opens a hidden window because OpenGL on Windows needs one, ignores the mouse, and destroys that window before returning. `run` is the fly loop. `render` is how a script or a test reads one picture.

Validation finishes before any window is opened and before any GPU allocation. On failure the call raises `ValueError` and returns. A second `run` after a finished flight is allowed and uses the same scene.

The binding rejects a scene when any of these is true.

- The soup has no triangles.
- An index does not name a vertex.
- A material index does not name a material.
- A normal has zero length.
- A position, a normal, an albedo, or a light value is not finite.
- An albedo component or a light intensity component is negative.
- A triangle's three positions are not three distinct points.
- The vertical field of view is less than or equal to 0, or greater than or equal to 180 degrees.
- The eye and the target are the same point.
- Up has zero length, or up is parallel to the view direction.

A normal that passes the check is scaled to unit length before the scene is accepted. The kernel trusts that length. The camera argument to `render` is checked with the same eye, target, up, and field-of-view rules, and that check also finishes before the hidden window opens.

After a scene is accepted, the rasterizer, the hierarchy builder, and the shade kernel do not check those conditions again.

The native stack is GLFW, OpenGL 4.5 core, CUDA 13, and pybind11. Python never sees a GL name or a CUDA pointer.

## Tests look at exceptions, pixels, and frame time

Tests need this GPU for `render` and for the frame-time check. Validation tests do not. They only need the exception.

A rejected scene raises `ValueError`, and the call returns. The cases are the list in the previous section. Callers branch on the exception type. They do not branch on the message text.

Color tests call `render` on a scene of large triangles placed so a chosen pixel is unambiguously one case. The expected color is computed in the test from the shade rules above. The test does not ask the engine for the expected color.

- A lit matte pixel equals albedo times 0.1, plus albedo times intensity times the clamped cosine.
- A shadowed matte pixel equals albedo times 0.1.
- A mirror pixel that reflects that lit matte surface equals the lit matte color.
- A raster miss equals 0.08, 0.09, 0.11.

The comparison allows an absolute error of 0.001 per channel.

The frame-time check decides whether the first version is done. It is not done if the median GPU time for raster plus shade, at 1920 by 1080 on this RTX 4050 Laptop, is above 16.7 milliseconds.

The timed scene is a room, one matte sphere, and one mirror sphere, with one point light. Each sphere is a UV sphere of 32 stacks and 32 slices. The timer is a GPU timestamp around raster and shade. Presenting to the monitor is outside that timestamp. Ten frames warm up. The next 100 frames produce a median and a spread. If the spread is wider than the gap between the median and 16.7 milliseconds, that run does not decide. A miss is followed by a profile of that same run. The limiter is whatever the profile shows.

## This version stops short of a second bounce

Objects do not move. There is one light. A reflection is a perfect mirror, one ray deep. There is no diffuse color bleed, no window resize, no OptiX, and no denoiser.
