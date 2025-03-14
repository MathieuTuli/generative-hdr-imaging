#define GL_SILENCE_DEPRECATION

#include <iostream>

#include "utils.hpp"

#include "imgui.h"
#include "math.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#define IM_CLAMP(V, MN, MX) ((V) < (MN) ? (MN) : (V) > (MX) ? (MX) : (V))

int G_WIDTH;
int G_HEIGHT;
GLFWwindow *G_WINDOW;
const char *GLSL_VERSION;
GLuint my_image_texture = 0;
int my_image_width = 0;
int my_image_height = 0;
ImVec4 CLEAR_COLOR = ImVec4(0.17f, 0.17f, 0.17f, 1.00f);

void loop() {
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetMainViewport()->Size.x - 12.f, 12.f), ImGuiCond_Once);
    ImGui::ShowDemoWindow();
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetMainViewport()->Size.x / 2.f, 12.f), ImGuiCond_Once);
    ImGui::ShowIDStackToolWindow();
    ImGui::SetNextWindowPos(
        ImVec2(ImGui::GetMainViewport()->Size.x / 2.f, 100.f), ImGuiCond_Once);
    ImGui::ShowMetricsWindow();

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoNav;
    ImGui::SetNextWindowPos(ImVec2(12.f, 12.f), ImGuiCond_Once);
    ImGui::SetNextWindowSize(ImVec2(300.f, 600.f), ImGuiCond_Once);
    // Create a window called "My First Tool", with a menu bar.
    bool my_tool_active;
    ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */
            }
            if (ImGui::MenuItem("Save", "Ctrl+S")) { /* Do stuff */
            }
            if (ImGui::MenuItem("Close", "Ctrl+W")) {
                my_tool_active = false;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }

    // Edit a color stored as 4 floats
    float my_color[4];
    ImGui::ColorEdit4("Color", my_color);

    // Generate samples and plot them
    float samples[100];
    for (int n = 0; n < 100; n++)
        samples[n] = sinf(n * 0.2f + ImGui::GetTime() * 1.5f);
    ImGui::PlotLines("Samples", samples, 100);

    // Display contents in a scrolling region
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Important Stuff");
    ImGui::BeginChild("Scrolling");
    for (int n = 0; n < 50; n++)
        ImGui::Text("%04d: Some text", n);
    ImGui::EndChild();
    ImGui::End();

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(G_WINDOW, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(CLEAR_COLOR.x * CLEAR_COLOR.w, CLEAR_COLOR.y * CLEAR_COLOR.w,
                 CLEAR_COLOR.z * CLEAR_COLOR.w, CLEAR_COLOR.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(G_WINDOW);
}

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int init_gl() {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }
    // We don't want the old OpenGL
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    GLSL_VERSION = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    GLSL_VERSION = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
#else
    // GL 3.0 + GLSL 130
    GLSL_VERSION = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
    // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

    int canvasWidth = G_WIDTH;
    int canvasHeight = G_HEIGHT;
    G_WINDOW = glfwCreateWindow(canvasWidth, canvasHeight, "float", NULL, NULL);
    if (G_WINDOW == NULL) {
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(G_WINDOW); // Initialize GLEW
    glfwSwapInterval(1);              // Enable vsync

    return 0;
}

int init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(G_WINDOW, true);
    ImGui_ImplOpenGL3_Init(GLSL_VERSION);
    PickUpAPencil();

    ImGuiIO &io = ImGui::GetIO();

    // Load Fonts
    io.Fonts->AddFontFromFileTTF("assets/poppins.ttf", 16.0f);
    io.Fonts->AddFontDefault();
    return 0;
}

int init() {
    init_gl();
    init_imgui();
    return 0;
}

void quit() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(G_WINDOW);
    glfwTerminate();
}

extern "C" int main(int argc, char **argv) {
    G_WIDTH = 1000;
    G_HEIGHT = 750;
    if (init() != 0)
        return 1;

    while (!glfwWindowShouldClose(G_WINDOW)) {
        loop();
    }
    quit();

    return 0;
}
