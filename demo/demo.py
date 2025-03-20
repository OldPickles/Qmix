import tkinter as tk
import time

# 创建主窗口
root = tk.Tk()
root.title("My Application")

# 添加一个标签
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

root.update()

print("Hello, Tkinter!")

time.sleep(5)

# 窗口关闭
root.withdraw()  # 隐藏窗口

print("Goodbye, Tkinter!")

# 启动主循环
root.mainloop()

