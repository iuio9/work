# 联邦学习菜单设置指南

## 问题说明

由于系统使用**动态路由模式** (`VITE_AUTH_ROUTE_MODE=dynamic`)，所有菜单都从数据库加载。
因此需要将联邦学习菜单添加到 `qczy_menu` 表才能在前端显示。

## 两种添加方式

### 方式一：使用 REST API（推荐）

如果后端服务正在运行，使用此方式最简单：

#### 1. 检查菜单是否已存在

```bash
curl http://localhost:9091/api/federated/init/check
```

#### 2. 添加联邦学习菜单

```bash
curl -X POST http://localhost:9091/api/federated/init/menu
```

**成功响应示例：**
```json
{
  "success": true,
  "message": "✅ 联邦学习菜单添加成功！请重新登录系统以加载新菜单",
  "menuExists": false,
  "menuInfo": {
    "id": 66,
    "menu_name": "federated-learning",
    "web_path": "/federated-learning",
    "icon": "carbon:machine-learning-model",
    "sort": 8
  }
}
```

#### 3. （可选）删除菜单（用于测试）

```bash
curl -X DELETE http://localhost:9091/api/federated/init/menu
```

### 方式二：直接执行 SQL 脚本

如果你有 MySQL 客户端访问权限：

#### 1. 使用 SQL 脚本

```bash
cd /home/user/work/back/datamark-admin
mysql -u root -pqczy1717 datamark < add_federated_learning_menu.sql
```

#### 2. 验证菜单是否添加成功

```bash
mysql -u root -pqczy1717 datamark -e "SELECT * FROM qczy_menu WHERE menu_name='federated-learning';"
```

#### 3. 或者使用检查脚本

```bash
mysql -u root -pqczy1717 datamark < check_federated_menu.sql
```

### 方式三：手动 SQL 命令

如果你更喜欢手动操作，可以直接在 MySQL 客户端执行：

```sql
USE datamark;

-- 插入联邦学习菜单
INSERT INTO `qczy_menu` VALUES (
  66, 0, 'federated-learning',
  'carbon:machine-learning-model',
  'carbon:machine-learning-model',
  NULL, 'route.federated-learning',
  '/federated-learning',
  'layout.base$view.federated-learning',
  NULL, 1, 8, NOW(), NOW(), 0, 0, NULL
);

-- 分配给管理员角色
INSERT INTO `qczy_role_menu` VALUES (1, 66);

-- 验证结果
SELECT * FROM qczy_menu WHERE menu_name='federated-learning';
```

## 添加菜单后的步骤

### 1. 清除浏览器缓存

按 `Ctrl + Shift + R` (Windows/Linux) 或 `Cmd + Shift + R` (Mac) 强制刷新浏览器

### 2. 重新登录系统

1. 退出登录
2. 使用 admin 账户重新登录
3. 系统会重新加载菜单权限

### 3. 验证菜单是否显示

登录后，在左侧菜单栏应该能看到：

```
📊 联邦学习  (Federated Learning)
```

图标：机器学习图标 (carbon:machine-learning-model)
排序：第 8 位（在 boxpulse 之后）

### 4. 访问联邦学习页面

点击菜单或直接访问：

```
http://localhost:8080/#/federated-learning
```

## 常见问题

### Q1: 点击菜单后仍然显示首页？

**原因：** 菜单未添加到数据库
**解决：** 按照上述方式添加菜单，然后重新登录

### Q2: 页面一直加载（转圈）？

**原因：** API 导出问题（已修复）或后端服务未启动
**解决：**
- 确保后端服务运行在 9091 端口
- 检查浏览器控制台是否有错误
- 确认 `src/service/api/index.ts` 包含 `export * from './federated';`

### Q3: 提示 403 或权限错误？

**原因：** 角色菜单关联未建立
**解决：** 确保执行了 `INSERT INTO qczy_role_menu VALUES (1, 66);`

### Q4: 提示菜单 ID 冲突？

**原因：** ID 66 已被占用
**解决：** 修改 SQL 中的 ID 为其他未使用的值（如 67, 68 等）

## 技术细节

### 菜单表结构

- **qczy_menu**: 存储所有菜单和页面定义
  - `id`: 菜单唯一标识
  - `parent_id`: 父菜单 ID（0 表示顶级菜单）
  - `menu_name`: 菜单名称（与前端路由对应）
  - `web_path`: URL 路径
  - `component`: Vue 组件路径
  - `sort`: 显示顺序

- **qczy_role_menu**: 角色与菜单的关联关系
  - `role_id`: 角色 ID（1 = admin）
  - `menu_id`: 菜单 ID

### 动态路由加载流程

1. 用户登录后，后端根据用户角色查询 `qczy_role_menu`
2. 获取该角色有权访问的所有 `menu_id`
3. 从 `qczy_menu` 表加载对应的菜单配置
4. 前端接收菜单数据，动态生成路由
5. 渲染侧边栏菜单

### REST API 实现位置

```
back/datamark-admin/src/main/java/com/qczy/federated/controller/FederatedMenuInitController.java
```

该控制器提供了三个端点：
- `POST /api/federated/init/menu`: 初始化菜单
- `GET /api/federated/init/check`: 检查菜单状态
- `DELETE /api/federated/init/menu`: 删除菜单（测试用）

## 下一步

菜单添加成功后，你可以：

1. **注册联邦学习节点**
2. **创建训练任务**
3. **启动联邦学习训练**
4. **监控训练进度和精度曲线**

详细使用说明请参考 `FEDERATED_LEARNING_README.md`
