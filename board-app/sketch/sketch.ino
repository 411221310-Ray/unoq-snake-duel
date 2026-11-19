// 貪食蛇 AI 對戰：MCU 端只負責畫 LED 矩陣。
// 畫面內容由 Linux 端（python/main.py、snake_core.py）決定，經 Bridge 呼叫 "draw" 送進來。
// 寫法照 Arduino 官方範例 core-and-foundational/02-led-matrix 與 led-matrix-painter。

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>
#include <zephyr/kernel.h>

Arduino_LED_Matrix matrix;

// Bridge 的 provider 跟 loop() 不在同一個執行緒，用 mutex 保護矩陣寫入
K_MUTEX_DEFINE(matrix_mtx);

const size_t FRAME_SIZE = 8 * 13;  // 8 列 × 13 欄

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // 每格亮度 0–7，和 snake_core.py 一致
  matrix.clear();

  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  delay(10);
}

// Python 端：Bridge.call("draw", <104 bytes>)
void draw(std::vector<uint8_t> frame) {
  if (frame.size() < FRAME_SIZE) return;

  k_mutex_lock(&matrix_mtx, K_FOREVER);
  matrix.draw(frame.data());
  k_mutex_unlock(&matrix_mtx);
}
