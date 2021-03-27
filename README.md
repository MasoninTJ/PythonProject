# 3D平台软件

随便写写，就当做是给过去的三年的一些总结

一些文档可通过我的Blog获得
https://www.yuque.com/zerosyujian/ucug9w

计划更新如下功能：

- [x] 几何类
  - [x] 点 `Point`
  - [x] 向量 `Vector`
  - [x] 线段 `Line`
  - [x] 平面 `Plane`
  - [x] 三角形 `Triangle`
  - [x] 网格 `Mesh`
  - [x] STL模型 `STLModel`
  - [x] 射线 `Ray3D`
  - [x] 包围盒 `BOX3D`
  - [x] 球 `Sphere`
  - [x] 圆 `Circle`
- [x] 点云读取显示
- [x] STL模型读取显示
- [ ] 鼠标左键按下旋转，track ball
- [ ] 鼠标滚轮缩放
- [ ] 鼠标中键平移
- [x] 键盘 escape 退出窗口
- [ ] 几何元素拟合算法
  - [x] 直线拟合
  - [x] 圆拟合
- [ ] 数值元素计算
- [ ] 最佳拟合

2021年3月25日11:55:52
- 解决绘制球标不能正确显示的问题，使用 `glPushMatrix` 与 `glPopMatrix`
- 区分 `glEnable(GL_DEPTH_TEST)` 与 `glDepthMask(GL_True)` 的区别
- 窗口如果不是正方形，显示的圆就不对，会显示为椭圆