uniform mat4   model;         // Model matrix
uniform mat4   rotation;          // View matrix
uniform mat4   translation;          // View matrix
uniform mat4   projection;    // Projection matrix
attribute vec3 position;      // Vertex position
uniform float t;
uniform float size;
uniform float amp;
uniform float k;
uniform float a_freq;
uniform float phase;
uniform vec4 color;
varying vec4 v_color;

vec3 wave_func(vec3 pos)
{
    // filled in the porgram preprocessing
      float d = sqrt(pos.x*pos.x + pos.z*pos.z);
      float y = amp * cos(d * k - a_freq * t + phase);
      return vec3(pos.x, y, pos.z);
    // ------------------------------------------
    return pos;
}

void main()
{
    vec4 pos = projection * translation * rotation * model * vec4(wave_func(position), 1.0);
    gl_Position = pos;
    v_color = color / pos.w * size;
}

