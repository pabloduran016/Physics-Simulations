uniform mat4   model;         // Model matrix
uniform mat4   rotation;          // View matrix
uniform mat4   translation;          // View matrix
uniform mat4   projection;    // Projection matrix
attribute vec3 position;      // Vertex position
uniform float t;
uniform float size;
uniform int n_waves;

uniform mat4 amps;
uniform mat4 ks;
uniform mat4 a_freqs;
uniform mat4 phases;
uniform float height;

//#define WAVES_CAP 25
//uniform float amps[WAVES_CAP];
//uniform float ks[WAVES_CAP];
//uniform float a_freqs[WAVES_CAP];
//uniform float phases[WAVES_CAP];

varying vec4 v_color;
uniform vec4 color;


#define amp 1. 
#define k 1.7951958
#define a_freq 0.0015708
#define phase 0.

vec3 wave_func(vec3 pos)
{
    if (pos.y < height) return pos;
    float d = sqrt(pos.x*pos.x + pos.z*pos.z);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            pos.y += amps[i][j] * cos(d * ks[i][j] - a_freqs[i][j] * t + phases[i][j]);
        }
    }
    return pos;
}

void main()
{
    gl_PointSize = 10;
    vec4 pos = projection * translation * rotation * model * vec4(wave_func(position), 1.0);
    gl_Position = pos;
    v_color = color / pos.w * size;
}

