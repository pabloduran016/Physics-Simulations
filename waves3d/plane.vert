uniform mat4 projection;
uniform mat4 translation;
uniform mat4 rotation;
uniform vec4 color;
uniform float scale;
varying vec4 v_color;
attribute vec3 position;


vec4 compute_postion(vec3 pos)
{
    pos.z = sign(pos.z) * scale;
    pos.x = sign(pos.x) * scale;
    return vec4(pos, 1.);
}

void main() {
    vec4 pos = compute_postion(position);
    gl_Position = projection * translation * rotation * pos;
    v_color = color / length(gl_Position) * scale;
}
