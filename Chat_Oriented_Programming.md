# Chat-Oriented Programming (CHOP)

## Goal
This document introduces the concept of Chat-Oriented Programming (CHOP), where developers leverage AI-assisted tools to enhance productivity. We will use Cursor AI IDE for demonstrations, but these concepts apply equally to VS Code (with Gemini Code Assist, GitHub Copilot, or relevant extensions) and Vim (using [vimlm](https://github.com/JosefAlbers/vimlm) or [vim-ai](https://github.com/madox2/vim-ai)).

### **OLD DAYS**
![Alt text](./img/RubberDuckProgramming.jpeg)

### **TO NEW WAYS** 

![Alt text](./img/CHOP.jpeg)


## Use Cases & Demonstrations

### **Use Case 1: Generate Boilerplate Code**

**Prompt in Cursor:**  
>"Can you write a code for gps driver ?"


### **Use Case 2: Refactor Code/Add new feature stepwise** 

**Prompts in Cursor:**  
>"Create a frontend module which can show near realtime location update on google map from the data from this driver,Use streamline for the UI."
>"Can you write code for fusion of realsense rgbd camera data with gps data for better localisation @Codebase"
>"Can you update the viewer with more details like showing rgbd data along with gps data and final fused view @Codebase"


### **Use Case 3 : Unit test** 

**Prompt in Cursor:**
>“Write pytest unit tests for @Codebase”

### **Use Case 4: Code Understanding and Documentation**

**Prompt in Cursor:**
>“Generate docstring documentation with explanation of the code @Codebase”

### **Use Case 5: Code Review** 

**Prompt in Cursor:**
>"Please do an in-depth code review such that defensive programing is followed ,  security guidelines are followed , For style guide followed python PEP8 @Codebase"


### **Use Case 6 : Try to add long extended feature request**

**Prompts in Cursor:**
>"Can you update the viewer with more details like showing rgbd data along with gps data and final fused view @Codebase"

>"Update the README.md with proper to use this codebase and also Dockerise the code @Codebase"






---

## Preferred Tools
- **VS Code** with:
  - [Gemini Code Assist](https://code.google.com/)
  - [GitHub Copilot](https://github.com/features/copilot)
- **Cursor IDE**
- **Vim** with:
  - [vimlm](https://github.com/JosefAlbers/vimlm)
  - [vim-ai](https://github.com/madox2/vim-ai)

---

This document serves as a hands-on guide for developers exploring AI-assisted coding workflows. Feel free to extend these examples as needed!
