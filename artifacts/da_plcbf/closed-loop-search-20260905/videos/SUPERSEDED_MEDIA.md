# Superseded and diagnostic media

`paced-collision-v1` and `paced-compensated-v1` use a -45 degree camera that obscures the fixed drone near the physical collision. They are superseded by the corresponding v2 render with a +225 degree camera and 4.3 m hover distance. The flight and contact states are unchanged. All videos stay local.

`camera-preview` contains camera selection diagnostics, including incomplete EGL rendering in a multi-scene preview process. These are not final media or additional experiments; they are excluded from publication. Final videos each render in a fresh EGL process and require nonblack scenes plus full decode verification.
