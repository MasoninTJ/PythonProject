# 对顶点着色器进行修改，顶点 + 向量 + 颜色
vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec3 a_color;

uniform mat4 model;  // 包含平移和旋转
uniform mat4 projection;
uniform mat4 view;

out vec3 v_color;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_color = a_color;
}
"""

# 修改片段着色器，去掉多余的输入与输出
fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
}
"""