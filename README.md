# MostDiffVal
AVS+ plugin
Returns sample with max absolute difference of clip1 and clip2 to reference clip. Support YUV 8-16 bits integer and float32. Planar YUV formats only.
All input clips must have equal format.

Params:

1. Clip Ref
2. Clip 1
3. Clip 2
4. threads: default 1. Enable internal multi-threading via OpenMP if > 1
5. opt: default -1 (auto). 0 - disabled SIMD, 1 - SSE 4.2, 2 - AVX2.
   
Usage:
MostDiffVal(clip_ref, clip1, clip2)

Where clip_ref recommended to be average of blurred clip1 and clip2.

Example of function GetSharpest:

Function GetSharpest(clip c1, clip c2)

{

avg=Layer(c1.GaussResize(c1.width, c1.height, src_left=0.001, src_top=0.001, p=2), c2.GaussResize(c2.width, c2.height, src_left=0.001, src_top=0.001, p=2), "fast")

return MostDiffVal(avg, c1, c2)

}

Expr() analog is Expr(Ref, clip1, clip2, "x y - abs x z - abs >= y z ?")
