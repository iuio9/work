/*
 Navicat Premium Dump SQL

 Source Server         : 127.0.0.1
 Source Server Type    : MySQL
 Source Server Version : 50719 (5.7.19)
 Source Host           : localhost:3306
 Source Schema         : datamark1

 Target Server Type    : MySQL
 Target Server Version : 50719 (5.7.19)
 File Encoding         : 65001

 Date: 21/08/2025 10:58:49
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for qczy_algorithm
-- ----------------------------
DROP TABLE IF EXISTS `qczy_algorithm`;
CREATE TABLE `qczy_algorithm`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `cur_task_id` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '当前任务Id',
  `model_id` varchar(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型id',
  `algorithm_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '算法名称',
  `algorithm_desc` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '算法描述',
  `url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '算法请求地址',
  `params` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '算法请求参数',
  `response_params` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '算法响应参数',
  `request_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '算法请求类型',
  `before_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '示意图处理前',
  `after_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '示意图处理后',
  `front_algorithm_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '前置功能',
  `train_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '训练类型 (0 正常 1 异常)',
  `ditcId` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '关联字段ID 如果参数值需要下拉选择',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 29 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_algorithm
-- ----------------------------
INSERT INTO `qczy_algorithm` VALUES (1, NULL, '1', '分割算法', '基于模型1实现的分割算法', 'http://192.168.1.4:5000/segment', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"image_path\",\"value\":\"\"},{\"type\":\"text\",\"label\":\"请输入前景点\",\"serverKey\":\"foreground_points\",\"value\":\"\"},{\"type\":\"text\",\"label\":\"请输入背景点\",\"serverKey\":\"background_points\",\"value\":\"\"}]', '[]', 'post', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (2, '25', '5', '风格转化-雾算法', '基于模型3的雾算法', 'http://192.168.1.4:5010/foggy', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"input_image\",\"value\":null,\"key\":\"input_image\",\"valuePlaceholder\":\"请选择图片路径\"},{\"key\":\"cfg_text\",\"value\":7.5,\"step\":\"0.5\",\"type\":\"text\",\"label\":\"请输入雾浓度\",\"sign\":\"number\",\"serverKey\":\"cfg_text\",\"valuePlaceholder\":\"请输入cfg_text\"},{\"key\":\"cfg_image\",\"value\":1.5,\"step\":\"0.5\",\"type\":\"text\",\"label\":\"请输入图片相似度\",\"sign\":\"number\",\"serverKey\":\"cfg_image\",\"valuePlaceholder\":\"请输入cfg_image\"}]', '[{\"type\":\"path\",\"label\":\"输出路径\",\"serverKey\":\"output_image\",\"value\":null,\"key\":\"output_image\",\"valuePlaceholder\":\"输出路径\"},{\"key\":\"original_image\",\"value\":null,\"type\":\"path\",\"label\":\"输出原图路径\",\"serverKey\":\"original_image\",\"valuePlaceholder\":\"输出原图路径\"},{\"type\":\"text\",\"label\":\"状态\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"状态\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/AntiBird_47.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/fog_AntiBird_47_gradual.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (3, '1713', '5', '风格转化-雪', '基于模型3的雪算法', 'http://192.168.1.4:5008/run_test', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"file_path\",\"value\":null,\"key\":\"file_path\",\"valuePlaceholder\":\"请选择图片路径\"},{\"type\":\"select\",\"label\":\"请选择模型名称\",\"value\":null,\"serverKey\":\"name\",\"options\":[{\"label\":\"tianjin_cyclegan模型\",\"value\":\"tianjin_cyclegan\"}],\"key\":\"name\",\"valuePlaceholder\":\"请选择模型名称\"},{\"type\":\"select\",\"label\":\"请选择模型类型\",\"serverKey\":\"model\",\"value\":null,\"options\":[{\"label\":\"雪模型-1\",\"value\":\"test\"}],\"key\":\"model\",\"valuePlaceholder\":\"请选择模型类型\"}]', '[{\"type\":\"path\",\"label\":\"返回路径\",\"serverKey\":\"results_path\",\"value\":null,\"key\":\"results_path\",\"valuePlaceholder\":\"返回路径\"},{\"type\":\"text\",\"label\":\"状态\",\"serverKey\":\"success\",\"value\":null,\"key\":\"success\",\"valuePlaceholder\":\"状态\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%AA/1727752228_real.png', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%AA/1727752228_fake.png', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (4, '7', '4', '数据增广-缺陷生成', '基于模型4的缺陷生成算法', 'http://192.168.1.4:5002/process_images', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"img_path\",\"value\":null,\"key\":\"img_path\",\"valuePlaceholder\":\"请选择图片路径\"},{\"type\":\"json\",\"label\":\"请选择json_path\",\"serverKey\":\"json_path\",\"value\":null,\"key\":\"json_path\",\"valuePlaceholder\":\"请选择json_path\"},{\"key\":\"option\",\"value\":\"\",\"type\":\"select\",\"label\":\"请选择生成缺陷目标\",\"serverKey\":\"option\"},{\"type\":\"text\",\"label\":\"请输入生成数量\",\"serverKey\":\"number\",\",value\":\"\",\"sign\":\"number\",\"key\":\"number\",\"value\":null,\"valuePlaceholder\":\"请输入生成数量\"}]', '[{\"type\":\"path\",\"label\":\"处理完成后的图片文件夹路径\",\"serverKey\":\"output_dir\",\"value\":null,\"key\":\"output_dir\",\"valuePlaceholder\":\"处理完成后的图片文件夹路径\"},{\"type\":\"text\",\"label\":\"成功信息\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"成功信息\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E7%BC%BA%E9%99%B7/%E7%BC%BA%E9%99%B7%E6%B5%8B%E8%AF%95-%E7%BB%9D%E7%BC%98%E5%AD%90.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E7%BC%BA%E9%99%B7/ef8af2ece15aca63f262b843021aff5.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (5, '1712', '4', '数据增广-异常区域生成', '基于模型4的异常区域生成算法', 'http://192.168.1.4:5001/gsam_inpainting', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"input_image\",\"value\":null,\"key\":\"input_image\",\"valuePlaceholder\":\"请选择图片路径\"},{\"type\":\"select\",\"label\":\"请输入要分割的对象名\",\"serverKey\":\"text_prompt1\",\"value\":null,\"key\":\"text_prompt1\",\"valuePlaceholder\":\"请输入要分割的对象名\"},{\"type\":\"select\",\"label\":\"请输入区域生成目标对象名\",\"serverKey\":\"text_prompt2\",\"value\":null,\"key\":\"text_prompt2\",\"valuePlaceholder\":\"请输入区域生成目标对象名\"},{\"type\":\"text\",\"label\":\"请输入随机选定区域的大小(根据需求自行调整,建议大小400-1500)\",\"serverKey\":\"area\",\"value\":null,\"key\":\"area\",\"valuePlaceholder\":\"请输入随机选定区域的大小(根据需求自行调整)\"}]', '[{\"type\":\"path\",\"label\":\"处理完成后的图片文件夹路径\",\"serverKey\":\"output_dir\",\"value\":null,\"key\":\"output_dir\",\"valuePlaceholder\":\"处理完成后的图片文件夹路径\"},{\"type\":\"text\",\"label\":\"成功信息\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"成功信息\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E5%8C%BA%E5%9F%9F/%E5%9C%BA%E6%99%AF%E7%94%9F%E6%88%90%E5%B7%A5%E7%A8%8B%E8%BD%A6%E8%BE%86.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E5%8C%BA%E5%9F%9F/%E5%B7%A5%E4%BD%9C%E4%B8%AD%E5%90%8A%E8%BD%A6.png', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (6, '12', '2', '元器件检测', '目标推理算法', 'http://192.168.1.4:5001/gsam_inpainting', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"value\":\"input_image\"},{\"type\":\"text\",\"label\":\"请输入要分割的对象名\",\"value\":\"text_prompt1\"},{\"type\":\"text\",\"label\":\"请输入区域生成目标对象名\",\"value\":\"text_prompt2\"},{\"type\":\"text\",\"label\":\"请输入随机选定区域的大小\",\"value\":\"area\"}]', '[{\"type\":\"path\",\"label\":\"处理完成后的图片文件夹路径\",\"value\":\"output_dir\"},{\"type\":\"text\",\"label\":\"成功信息\",\"value\":\"status\"}]', 'post', '', '', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (7, NULL, '2', '异常检测', '目标推理算法', 'http://192.168.1.4:5001/gsam_inpainting', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"value\":\"input_image\"},{\"type\":\"text\",\"label\":\"请输入要分割的对象名\",\"value\":\"text_prompt1\"},{\"type\":\"text\",\"label\":\"请输入区域生成目标对象名\",\"value\":\"text_prompt2\"},{\"type\":\"text\",\"label\":\"请输入随机选定区域的大小\",\"value\":\"area\"}]', '[{\"type\":\"path\",\"label\":\"处理完成后的图片文件夹路径\",\"value\":\"output_dir\"},{\"type\":\"text\",\"label\":\"成功信息\",\"value\":\"status\"}]', 'post', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (8, '1711', '5', '风格转化-烟火算法', '基于模型3的烟火算法', 'http://192.168.1.4:5011/smoke', '[{\"type\":\"image\",\"label\":\"请选择图片路径\",\"serverKey\":\"input_image\",\"value\":null,\"key\":\"input_image\",\"valuePlaceholder\":\"请选择图片路径\"},{\"key\":\"cfg_text\",\"value\":7.5,\"step\":\"0.5\",\"type\":\"text\",\"label\":\"请输入烟火浓度\",\"sign\":\"number\",\"serverKey\":\"cfg_text\",\"valuePlaceholder\":\"请输入雾浓度\"},{\"key\":\"cfg_image\",\"value\":1.5,\"step\":\"0.5\",\"type\":\"text\",\"label\":\"请输入图片相似度\",\"sign\":\"number\",\"serverKey\":\"cfg_image\",\"valuePlaceholder\":\"请输入图片相似度\"}]', '[{\"type\":\"path\",\"label\":\"输出路径\",\"serverKey\":\"output_image\",\"value\":null,\"key\":\"output_image\",\"valuePlaceholder\":\"输出路径\"},{\"key\":\"original_image\",\"value\":null,\"type\":\"path\",\"label\":\"输出原图路径\",\"serverKey\":\"original_image\",\"valuePlaceholder\":\"输出原图路径\"},{\"type\":\"text\",\"label\":\"状态\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"状态\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/AntiBird_47.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/fog_AntiBird_47_gradual.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (9, '12', '3', '目标检测', '目标检测算法', 'http://10.5.28.222:5005/process_images', '[{\"type\":\"image\",\"label\":\"请输入单个图片路径\",\"serverKey\":\"img_url\",\"value\":null,\"key\":\"img_url\",\"valuePlaceholder\":\"请输入单个图片路径\"},{\"key\":\"weight\",\"value\":null,\"type\":\"select\",\"label\":\"请选择模型\",\"serverKey\":\"weight\",\"valuePlaceholder\":\"请选择模型\"},{\"key\":\"mode\",\"value\":\"0\",\"type\":\"text\",\"isShow\":false,\"label\":\"选择检测任务\",\"serverKey\":\"mode\",\"valuePlaceholder\":\"选择检测任务\"},{\"key\":\"version\",\"value\":\"\",\"type\":\"select\",\"label\":\"选择模型版本\",\"serverKey\":\"version\"}]', '[{\"type\":\"path\",\"label\":\"输出处理结束后路径\",\"serverKey\":\"output_image\",\"value\":null,\"key\":\"output_image\",\"valuePlaceholder\":\"输出处理结束后路径\"},{\"key\":\"original_image\",\"value\":null,\"type\":\"path\",\"label\":\"输出原图路径\",\"serverKey\":\"original_image\",\"valuePlaceholder\":\"输出原图路径\"},{\"type\":\"path\",\"label\":\"label文件\",\"serverKey\":\"label_json\",\"value\":null,\"key\":\"label_json\",\"valuePlaceholder\":\"label文件\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/AntiBird_47.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/fog_AntiBird_47_gradual.jpg', NULL, '0', NULL);
INSERT INTO `qczy_algorithm` VALUES (10, '10', '3', '元器件检测', '异常检测模型', 'http://10.5.28.222:5005/process_images', '[{\"type\":\"image\",\"label\":\"请输入单个图片路径\",\"serverKey\":\"img_url\",\"value\":null,\"key\":\"img_url\",\"valuePlaceholder\":\"请输入单个图片路径\"},{\"key\":\"weight\",\"value\":null,\"type\":\"select\",\"label\":\"请选择模型\",\"serverKey\":\"weight\",\"valuePlaceholder\":\"请选择模型\"},{\"key\":\"mode\",\"value\":\"1\",\"isShow\":false,\"type\":\"text\",\"label\":\"选择检测任务\",\"serverKey\":\"mode\",\"valuePlaceholder\":\"选择检测任务\"},{\"key\":\"version\",\"value\":\"\",\"type\":\"select\",\"label\":\"请选择模型版本\",\"serverKey\":\"version\"}]', '[{\"type\":\"path\",\"label\":\"输出处理结束后路径\",\"serverKey\":\"output_image\",\"value\":null,\"key\":\"output_image\",\"valuePlaceholder\":\"输出处理结束后路径\"},{\"key\":\"original_image\",\"value\":null,\"type\":\"path\",\"label\":\"输出原图路径\",\"serverKey\":\"original_image\",\"valuePlaceholder\":\"输出原图路径\"},{\"type\":\"path\",\"label\":\"label文件\",\"serverKey\":\"label_json\",\"value\":null,\"key\":\"label_json\",\"valuePlaceholder\":\"label文件\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/AntiBird_47.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E9%9B%BE/fog_AntiBird_47_gradual.jpg', NULL, '1', NULL);
INSERT INTO `qczy_algorithm` VALUES (11, '1866', '6', '数据集下载', '训练前置功能-数据集下载', 'http://10.5.28.222:5006/download_images', '[{\"type\":\"text\",\"label\":\"请选择图片路径\",\"serverKey\":\"img_path\",\"value\":null,\"key\":\"img_path\",\"valuePlaceholder\":\"请选择图片路径\"},{\"type\":\"text\",\"label\":\"请选择json_path\",\"serverKey\":\"json_path\",\"value\":null,\"key\":\"json_path\",\"valuePlaceholder\":\"请选择json_path\"}]', '[{\"type\":\"text\",\"label\":\"成功信息\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"成功信息\"}]', 'post', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E7%BC%BA%E9%99%B7/%E7%BC%BA%E9%99%B7%E6%B5%8B%E8%AF%95-%E7%BB%9D%E7%BC%98%E5%AD%90.jpg', 'http://192.168.1.3:9092/formal/demo/%E7%94%B5%E5%8A%9B-%E6%B5%8B%E8%AF%95%E6%BC%94%E7%A4%BA%E5%9B%BE%E7%89%87%281%29/%E7%BC%BA%E9%99%B7/ef8af2ece15aca63f262b843021aff5.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (12, '1872', '7', '模型预处理', '模型训练前置功能', 'http://10.5.28.222:5007/process_images', '[{\"type\":\"text\",\"label\":\"检测任务\",\"serverKey\":\"mode\",\"value\":null,\"key\":\"mode\",\"valuePlaceholder\":\"检测任务\"},{\"type\":\"text\",\"label\":\"所有数据集标签\",\"serverKey\":\"classes\",\"value\":null,\"key\":\"classes\",\"valuePlaceholder\":\"所有标签\"}]', '[{\"type\":\"text\",\"label\":\"成功信息\",\"serverKey\":\"status\",\"value\":null,\"key\":\"status\",\"valuePlaceholder\":\"成功信息\"}]', 'post', NULL, NULL, '11', NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (13, '1872', '8', '模型训练', '模型训练功能', 'http://10.5.28.222:5008/train_images', '[{\"key\":\"epochs\",\"value\":\"300\",\"type\":\"text\",\"valuePlaceholder\":\"训练轮数(epochs)\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"训练轮数(epochs)\",\"serverKey\":\"epochs\",\"tooltip\":\"看loss曲线，如果train loss和val loss都还有下降空间，就继续加大epoch,如果基本平了，加大epoch用处也不大了，如果train loss降val loss降着降着上升了，这说明，模型在val loss由降转升的转折点就收敛了。\"},{\"key\":\"weights\",\"value\":\"s\",\"type\":\"text\",\"label\":\"预训练权重文件(weights)\",\"serverKey\":\"weights\",\"valuePlaceholder\":\"预训练权重文件(weights)\",\"tooltip\":\"按S，M，L，X的顺序，模型参数量逐级增加，准确度逐级增加，训练和推理时间逐级增加，GPU资源消耗逐级增加。\"},{\"key\":\"batch_size\",\"value\":\"32\",\"type\":\"text\",\"label\":\"训练批次大小(batch_size)\",\"serverKey\":\"batch_size\",\"valuePlaceholder\":\"训练批次大小(batch_size)\",\"tooltip\":\"Batch size 的选择与可用 GPU 显存直接相关。如果 batch size 设置得太大，可能会导致内存溢出（Out of Memory）。可以通过观察 GPU 使用情况来调整 batch size。通常，使用更大的 batch size 可以加速训练，因为每个训练步骤中会并行处理更多数据，但也需要更多显存。在 GPU 显存允许的情况下，较大的 batch size（如 32、64、128 等）可以加速训练，因为它们使得每次迭代的计算量更大，但计算图和反向传播可以同时处理更多样本。小的 batch size（如 4、8、16）通常需要更多的训练步骤才能完成一次训练周期，但它们通常能提供更精细的梯度估计，并且可能有助于避免一些局部最优解。较大的 batch size 可以减少训练过程中的梯度波动，通常会带来更稳定的训练过程，但可能会导致模型更容易过拟合，因为它的梯度估计较为精确。较小的 batch size 可以提供更多的随机性，可能对模型的泛化能力更有帮助，但可能会导致训练过程中的噪声较大。\"},{\"key\":\"number_classes\",\"value\":\"3\",\"type\":\"text\",\"label\":\"类别总数\",\"bindKey\":\"tagNum\",\"serverKey\":\"number_classes\",\"valuePlaceholder\":\"类别总数\"},{\"key\":\"name_classes\",\"value\":\"damper,insulator,foreignmatter,windbirdrepellent\",\"type\":\"text\",\"label\":\"类别名称(英文逗号隔开)\",\"bindKey\":\"tags\",\"serverKey\":\"name_classes\",\"valuePlaceholder\":\"类别名称(英文逗号隔开)\"},{\"key\":\"img_size\",\"value\":\"640\",\"type\":\"text\",\"label\":\"图像尺寸(宽和高)\",\"serverKey\":\"img_size\",\"valuePlaceholder\":\"图像尺寸(宽和高)\",\"tooltip\":\"调节经验：YOLOv5 的默认图像尺寸是 640x640，这是一个在精度和效率之间取得较好平衡的尺寸。硬件限制： 如果 GPU 显存有限，可能需要选择较小的图像尺寸（如 320x320 或 416x416），以避免显存溢出。较小的图像尺寸也能加速训练过程。检测精度要求： 如果任务要求较高的精度，特别是在检测小物体时，较大的图像尺寸（如 640x640 或 1024x1024）可能更合适，因为它能够保留更多细节信息。数据的特性： 如果数据中的目标物体较大或者背景较为简单，使用较小的图像尺寸（如 320x320）可能不会影响模型的性能太多，反而有助于提高训练效率。\"},{\"key\":\"image_weights\",\"value\":\"false\",\"type\":\"text\",\"label\":\"是否使用图像权重\",\"serverKey\":\"image_weights\",\"valuePlaceholder\":\"是否使用图像权重\",\"tooltip\":\"调节经验：当不同类别的图片数量相差较大时，可使用图像权重，以平衡每类数据训练时的学习力度。但还是建议原始数据中类别数量均衡。\"},{\"key\":\"name\",\"value\":\"gdp\",\"type\":\"text\",\"label\":\"模型名称\",\"serverKey\":\"name\",\"valuePlaceholder\":\"任务名称\",\"tooltip\":\"模型名称,如果重复会覆盖原有模型\"},{\"key\":\"epochs_resnet\",\"value\":\"100\",\"type\":\"text\",\"label\":\"元器件(异常)检测-epochs\",\"serverKey\":\"epochs_resnet\",\"valuePlaceholder\":\"元器件(异常)检测-epochs\",\"tooltip\":\"调节经验：如果数据集较小或者较简单，30-50 epochs 可能就足够了。对于更复杂或更大的数据集，可能需要 50-100 epochs 甚至更多，尤其是在初始训练时，学习率较低，训练可能需要更多轮次来收敛。\"},{\"key\":\"batch_size_resnet\",\"value\":\"32\",\"type\":\"text\",\"label\":\"元器件(异常)检测-batch_size\",\"serverKey\":\"batch_size_resnet\",\"valuePlaceholder\":\"元器件(异常)检测-batch_size\",\"tooltip\":\"调节经验：较大的 batch size 能在每次迭代中并行处理更多数据，通常会加速训练。如果显存较小，或者在训练过程中遇到 Out Of Memory (OOM) 错误，可以考虑减小为 16 或更小的 batch size。Batch size 较小时，训练过程可能更为稳定，但每个 epoch 需要的时间更长。\"},{\"key\":\"learning_rate\",\"value\":\"0.01\",\"type\":\"text\",\"label\":\"元器件(异常)检测学习率\",\"serverKey\":\"learning_rate\",\"valuePlaceholder\":\"元器件(异常)检测学习率\",\"tooltip\":\"调节经验：如果学习率设置得太高，训练可能会震荡，甚至无法收敛。如果学习率太低，训练会变得过慢。此时可以尝试逐步增加学习率，直到训练速度和稳定性都达到平衡。\"},{\"key\":\"mode\",\"value\":null,\"type\":\"text\",\"label\":\"mode\",\"serverKey\":\"mode\",\"valuePlaceholder\":\"mode\",\"isShow\":\"false\"},{\"key\":\"index\",\"value\":null,\"type\":\"text\",\"label\":\"index\",\"serverKey\":\"index\",\"isShow\":\"false\",\"valuePlaceholder\":\"index\"}]', '[{\"key\":\"wandb\",\"value\":null,\"type\":\"text\",\"label\":\"http路径\",\"serverKey\":\"wandb\",\"valuePlaceholder\":\"http路径\"},{\"key\":\"pid\",\"value\":null,\"type\":\"text\",\"label\":\"目标检测训练进程id\",\"serverKey\":\"pid\",\"valuePlaceholder\":\"目标检测训练进程id\"}]', 'post', NULL, NULL, '11', NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (14, NULL, '-1', '结束训练', '结束训练', 'http://10.5.28.222:5009/pid_kill', '[{\"key\":\"pid_list\",\"value\":\"\",\"type\":\"text\",\"valuePlaceholder\":\"请输入参数Value\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"pid_list\",\"serverKey\":\"pid_list\"}]', '[{\"key\":\"status\",\"value\":\"\",\"type\":\"text\",\"valuePlaceholder\":\"请输入参数Value\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"status\",\"serverKey\":\"status\"}]', 'post', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (15, NULL, '-1', '模型评估', '模型评估', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"weight\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"weight\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"weight\",\"serverKey\":\"weight\"}]', '[{\"key\":\"model_path\",\"value\":null,\"type\":\"path\",\"valuePlaceholder\":\"model path\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"model_path\",\"serverKey\":\"model_path\"}]', 'post', NULL, NULL, NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (16, '24', '9', '尺寸变化', '数据增强算子-尺寸变化', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"width\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"width\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入尺寸变化宽度\",\"serverKey\":\"width\"},{\"key\":\"height\",\"value\":null,\"type\":\"text\",\"label\":\"请输入尺寸变化高度\",\"serverKey\":\"height\",\"valuePlaceholder\":\"height\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/%E5%B0%BA%E5%AF%B8%E5%8F%98%E5%8C%96.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (17, NULL, '9', '水平镜像', '数据增强算子-水平镜像', 'http://10.5.28.222:5010/assess_images', '', '', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/%E6%B0%B4%E5%B9%B3%E9%95%9C%E5%83%8F.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (18, NULL, '9', '随机区域剪裁', '数据增强算子-随机区域剪裁', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"widht\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"widht\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入区域采集宽度\",\"serverKey\":\"widht\"},{\"key\":\"height\",\"value\":null,\"type\":\"text\",\"label\":\"请输入区域采集高度\",\"serverKey\":\"height\",\"valuePlaceholder\":\"height\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/随机区域剪裁.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (19, NULL, '9', '中心剪裁', '数据增强算子-中心剪裁', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"width\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"width\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入裁剪尺寸宽度\",\"serverKey\":\"width\"},{\"key\":\"height\",\"value\":null,\"type\":\"text\",\"label\":\"请输入裁剪尺寸高度\",\"serverKey\":\"height\",\"valuePlaceholder\":\"height\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/中心剪裁.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (20, NULL, '9', '灰度化', '数据增强算子-灰度化', 'http://10.5.28.222:5010/assess_images', '', '', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/灰度化.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (21, NULL, '9', '仿射变化', '数据增强算子-仿射变化', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"scale\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"scale\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入放射参数scale\",\"serverKey\":\"scale\"},{\"key\":\"rotate\",\"value\":null,\"type\":\"text\",\"label\":\"请输入放射参数rotate\",\"serverKey\":\"rotate\",\"valuePlaceholder\":\"rotate\"},{\"key\":\"shear\",\"value\":null,\"type\":\"text\",\"label\":\"请输入放射参数shear\",\"serverKey\":\"shear\",\"valuePlaceholder\":\"shear\"},{\"key\":\"translate_percent\",\"value\":null,\"type\":\"text\",\"label\":\"请输入放射参数translate_percent\",\"serverKey\":\"translate_percent\",\"valuePlaceholder\":\"translate_percent\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/仿射变化.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (22, NULL, '9', '平移缩放旋转', '数据增强算子-平移缩放旋转', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"shift_limit\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"shift_limit\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入平移参数scale_limit\",\"serverKey\":\"shift_limit\"},{\"key\":\"scale_limit\",\"value\":null,\"type\":\"text\",\"label\":\"请输入缩放参数scale_limit\",\"serverKey\":\"scale_limit\",\"valuePlaceholder\":\"scale_limit\"},{\"key\":\"rotate_limit\",\"value\":null,\"type\":\"text\",\"label\":\"请输入旋转参数rotate_limit\",\"serverKey\":\"rotate_limit\",\"valuePlaceholder\":\"rotate_limit\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/平移缩放旋转.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (23, NULL, '9', 'RGB偏移', 'RGB偏移-平移缩放旋转', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"r_shift_limit\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"r_shift_limit\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入偏移参数r_shift_limit\",\"serverKey\":\"r_shift_limit\"},{\"key\":\"g_shift_limit\",\"value\":null,\"type\":\"text\",\"label\":\"请输入偏移参数g_shift_limit\",\"serverKey\":\"g_shift_limit\",\"valuePlaceholder\":\"g_shift_limit\"},{\"key\":\"b_shift_limit\",\"value\":null,\"type\":\"text\",\"label\":\"请输入偏移参数b_shift_limit\",\"serverKey\":\"b_shift_limit\",\"valuePlaceholder\":\"b_shift_limit\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/RGB偏移.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (24, NULL, '9', '运动模糊', '数据增强算子-运动模糊', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"blur_limit\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"blur_limit\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入模糊参数blur_limit\",\"serverKey\":\"blur_limit\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/运动模糊.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (25, NULL, '9', '随机亮度对比', '数据增强算子-随机亮度对比', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"brightness_limit\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"brightness_limit\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入亮度参数(最大变化 20%)\",\"serverKey\":\"brightness_limit\"},{\"key\":\"contrast_limit\",\"value\":null,\"type\":\"text\",\"label\":\"请输入对比度参数(最大变化 20%)\",\"serverKey\":\"contrast_limit\",\"valuePlaceholder\":\"contrast_limit\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/随机亮度对比.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (26, NULL, '9', '锐化', '数据增强算子-锐化', 'http://10.5.28.222:5010/assess_images', '[{\"key\":\"alpha\",\"value\":null,\"type\":\"text\",\"valuePlaceholder\":\"alpha\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"请输入锐化参数(11)锐化参数范围 (0.5, 1.0)\",\"serverKey\":\"alpha\"}]', '[]', 'post', 'http://121.41.225.75:7091/formal/examplePic/AntiBird_37.jpg', 'http://121.41.225.75:7091/formal/examplePic/锐化.jpg', NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (27, NULL, NULL, '123', '12321', NULL, '[]', '[]', NULL, NULL, NULL, NULL, NULL, NULL);
INSERT INTO `qczy_algorithm` VALUES (28, NULL, NULL, '输电', NULL, NULL, '[]', '[]', NULL, NULL, NULL, NULL, NULL, NULL);

-- ----------------------------
-- Table structure for qczy_algorithm_model
-- ----------------------------
DROP TABLE IF EXISTS `qczy_algorithm_model`;
CREATE TABLE `qczy_algorithm_model`  (
  `model_id` int(11) NOT NULL AUTO_INCREMENT,
  `model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '算法模型名称',
  `model_biz_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型类型及业务标签',
  `model_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型接口地址',
  `model_params` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '模型参数',
  `model_env_params` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '模型环境参数',
  `model_req_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '请求方式',
  `model_desc` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型描述信息',
  `is_delete` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '是否可删除',
  `train_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型训练过程提供了一个链接地址(可视化)',
  `train_console` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '训练终端->查看日志输出',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `train_stat` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '训练状态',
  `train_task_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '当前训练任务id',
  `assess_lst` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '评估列表',
  `model_version` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型版本',
  PRIMARY KEY (`model_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 9 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_algorithm_model
-- ----------------------------
INSERT INTO `qczy_algorithm_model` VALUES (3, 'best模型', '该模型包含目标检测和异常检测任务', '/app/runs/train/ghf', '[{\"key\":\"epochs\",\"value\":\"300\",\"type\":\"text\",\"valuePlaceholder\":\"训练轮数(epochs)\",\"keyPlaceholder\":\"请输入参数Key\",\"label\":\"训练轮数(epochs)\",\"serverKey\":\"epochs\",\"tooltip\":\"看loss曲线，如果train loss和val loss都还有下降空间，就继续加大epoch,如果基本平了，加大epoch用处也不大了，如果train loss降val loss降着降着上升了，这说明，模型在val loss由降转升的转折点就收敛了。\"},{\"key\":\"weights\",\"value\":\"s\",\"type\":\"text\",\"label\":\"预训练权重文件(weights)\",\"serverKey\":\"weights\",\"valuePlaceholder\":\"预训练权重文件(weights)\",\"tooltip\":\"按S，M，L，X的顺序，模型参数量逐级增加，准确度逐级增加，训练和推理时间逐级增加，GPU资源消耗逐级增加。\"},{\"key\":\"batch_size\",\"value\":\"32\",\"type\":\"text\",\"label\":\"训练批次大小(batch_size)\",\"serverKey\":\"batch_size\",\"valuePlaceholder\":\"训练批次大小(batch_size)\",\"tooltip\":\"Batch size 的选择与可用 GPU 显存直接相关。如果 batch size 设置得太大，可能会导致内存溢出（Out of Memory）。可以通过观察 GPU 使用情况来调整 batch size。通常，使用更大的 batch size 可以加速训练，因为每个训练步骤中会并行处理更多数据，但也需要更多显存。在 GPU 显存允许的情况下，较大的 batch size（如 32、64、128 等）可以加速训练，因为它们使得每次迭代的计算量更大，但计算图和反向传播可以同时处理更多样本。小的 batch size（如 4、8、16）通常需要更多的训练步骤才能完成一次训练周期，但它们通常能提供更精细的梯度估计，并且可能有助于避免一些局部最优解。较大的 batch size 可以减少训练过程中的梯度波动，通常会带来更稳定的训练过程，但可能会导致模型更容易过拟合，因为它的梯度估计较为精确。较小的 batch size 可以提供更多的随机性，可能对模型的泛化能力更有帮助，但可能会导致训练过程中的噪声较大。\"},{\"key\":\"number_classes\",\"value\":\"3\",\"type\":\"text\",\"label\":\"类别总数\",\"bindKey\":\"tagNum\",\"serverKey\":\"number_classes\",\"valuePlaceholder\":\"类别总数\"},{\"key\":\"name_classes\",\"value\":\"damper,insulator,foreignmatter,windbirdrepellent\",\"type\":\"text\",\"label\":\"类别名称(英文逗号隔开)\",\"bindKey\":\"tags\",\"serverKey\":\"name_classes\",\"valuePlaceholder\":\"类别名称(英文逗号隔开)\"},{\"key\":\"img_size\",\"value\":\"640\",\"type\":\"text\",\"label\":\"图像尺寸(宽和高)\",\"serverKey\":\"img_size\",\"valuePlaceholder\":\"图像尺寸(宽和高)\",\"tooltip\":\"调节经验：YOLOv5 的默认图像尺寸是 640x640，这是一个在精度和效率之间取得较好平衡的尺寸。硬件限制： 如果 GPU 显存有限，可能需要选择较小的图像尺寸（如 320x320 或 416x416），以避免显存溢出。较小的图像尺寸也能加速训练过程。检测精度要求： 如果任务要求较高的精度，特别是在检测小物体时，较大的图像尺寸（如 640x640 或 1024x1024）可能更合适，因为它能够保留更多细节信息。数据的特性： 如果数据中的目标物体较大或者背景较为简单，使用较小的图像尺寸（如 320x320）可能不会影响模型的性能太多，反而有助于提高训练效率。\"},{\"key\":\"image_weights\",\"value\":\"false\",\"type\":\"text\",\"label\":\"是否使用图像权重\",\"serverKey\":\"image_weights\",\"valuePlaceholder\":\"是否使用图像权重\",\"tooltip\":\"调节经验：当不同类别的图片数量相差较大时，可使用图像权重，以平衡每类数据训练时的学习力度。但还是建议原始数据中类别数量均衡。\"},{\"key\":\"name\",\"value\":\"gdp\",\"type\":\"text\",\"label\":\"模型名称\",\"serverKey\":\"name\",\"valuePlaceholder\":\"任务名称\",\"tooltip\":\"模型名称,如果重复会覆盖原有模型\"},{\"key\":\"epochs_resnet\",\"value\":\"100\",\"type\":\"text\",\"label\":\"元器件(异常)检测-epochs\",\"serverKey\":\"epochs_resnet\",\"valuePlaceholder\":\"元器件(异常)检测-epochs\",\"tooltip\":\"调节经验：如果数据集较小或者较简单，30-50 epochs 可能就足够了。对于更复杂或更大的数据集，可能需要 50-100 epochs 甚至更多，尤其是在初始训练时，学习率较低，训练可能需要更多轮次来收敛。\"},{\"key\":\"batch_size_resnet\",\"value\":\"32\",\"type\":\"text\",\"label\":\"元器件(异常)检测-batch_size\",\"serverKey\":\"batch_size_resnet\",\"valuePlaceholder\":\"元器件(异常)检测-batch_size\",\"tooltip\":\"调节经验：较大的 batch size 能在每次迭代中并行处理更多数据，通常会加速训练。如果显存较小，或者在训练过程中遇到 Out Of Memory (OOM) 错误，可以考虑减小为 16 或更小的 batch size。Batch size 较小时，训练过程可能更为稳定，但每个 epoch 需要的时间更长。\"},{\"key\":\"learning_rate\",\"value\":\"0.01\",\"type\":\"text\",\"label\":\"元器件(异常)检测学习率\",\"serverKey\":\"learning_rate\",\"valuePlaceholder\":\"元器件(异常)检测学习率\",\"tooltip\":\"调节经验：如果学习率设置得太高，训练可能会震荡，甚至无法收敛。如果学习率太低，训练会变得过慢。此时可以尝试逐步增加学习率，直到训练速度和稳定性都达到平衡。\"},{\"key\":\"mode\",\"value\":null,\"type\":\"text\",\"label\":\"mode\",\"serverKey\":\"mode\",\"valuePlaceholder\":\"mode\",\"isShow\":\"false\"},{\"key\":\"index\",\"value\":null,\"type\":\"text\",\"label\":\"index\",\"serverKey\":\"index\",\"isShow\":\"false\",\"valuePlaceholder\":\"index\"}]', '', NULL, '该模型包含目标检测和异常检测任务', NULL, '', NULL, '2024-11-01 17:46:33', '2024-11-26 13:47:13', NULL, NULL, '2165,2164,2163,2162', NULL);
INSERT INTO `qczy_algorithm_model` VALUES (8, 'gdp', NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

-- ----------------------------
-- Table structure for qczy_algorithm_task
-- ----------------------------
DROP TABLE IF EXISTS `qczy_algorithm_task`;
CREATE TABLE `qczy_algorithm_task`  (
  `task_id` int(11) NOT NULL AUTO_INCREMENT,
  `model_id` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务绑定模型id',
  `data_set_id` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务输入数据集id',
  `version` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '数据集id版本',
  `dataset_out_id` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务输出数据集id',
  `algorithm_id` varchar(30) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '绑定模型id',
  `task_name` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务名称',
  `task_stat` varchar(2550) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '（开始、进行中、结束、异常）',
  `task_exception` text CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '如果状态为异常该字段不为Null',
  `task_progress` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务百分比',
  `is_train` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '0' COMMENT '是否训练',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `task_input_name` varchar(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务输入名称',
  `task_desc` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '任务描述',
  `train_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '训练地址',
  `train_console` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL,
  `pid` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT 'pid',
  `assess_url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '' COMMENT '新模型地址',
  `train_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '训练功能(0 目标检测  1.异常检测)',
  `is_assess` varchar(255) CHARACTER SET tis620 COLLATE tis620_thai_ci NULL DEFAULT NULL COMMENT '是否在评估中(0,未评估,1. 已经评估2. 正在评估)',
  `record_type` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT '0' COMMENT '记录类型',
  `model_version` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型版本',
  `model_name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL DEFAULT NULL COMMENT '模型名称',
  PRIMARY KEY (`task_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci COMMENT = '算法任务表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_algorithm_task
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_algorithm_task_result
-- ----------------------------
DROP TABLE IF EXISTS `qczy_algorithm_task_result`;
CREATE TABLE `qczy_algorithm_task_result`  (
  `task_id` bigint(20) NOT NULL,
  `task_params` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '算法任务参数',
  `task_result` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NULL COMMENT '算法任务返回的数据',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间'
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_algorithm_task_result
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_button_permission
-- ----------------------------
DROP TABLE IF EXISTS `qczy_button_permission`;
CREATE TABLE `qczy_button_permission`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `menu_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '菜单名称',
  `parent_id` int(11) NULL DEFAULT NULL COMMENT '父id',
  `button_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '按钮名称',
  `permission` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '权限',
  `sort` int(8) NULL DEFAULT NULL COMMENT '排序',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 132 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_button_permission
-- ----------------------------
INSERT INTO `qczy_button_permission` VALUES (101, '多人标注', 16, '管理多人标注团队', 'system:mul:manage', 1);
INSERT INTO `qczy_button_permission` VALUES (102, '多人标注', 16, '创建任务', 'system:mul:createTask', 2);
INSERT INTO `qczy_button_permission` VALUES (103, '测试评估申请', 56, '查看详情', 'thirdparty:mul:details', 1);
INSERT INTO `qczy_button_permission` VALUES (104, '测试评估申请', 56, '模型调试', 'thirdparty:mul:modelDebugging', 2);
INSERT INTO `qczy_button_permission` VALUES (105, '测试评估申请', 56, '提交审核', 'thirdparty:mul:submitForReview', 3);
INSERT INTO `qczy_button_permission` VALUES (106, '测试评估申请', 56, '生成数据附件', 'thirdparty:mul:generateAttachments', 4);
INSERT INTO `qczy_button_permission` VALUES (107, '测试评估申请', 56, '上传数据附件', 'thirdparty:mul:uploadAttachments', 5);
INSERT INTO `qczy_button_permission` VALUES (108, '测试评估审批', 57, '查看详情', 'thirdparty:approve:detail', 1);
INSERT INTO `qczy_button_permission` VALUES (109, '测试评估审批', 57, '审批通过', 'thirdparty:approve:approve', 2);
INSERT INTO `qczy_button_permission` VALUES (110, '测试评估审批', 57, '审批退回', 'thirdparty:approve:reject', 3);
INSERT INTO `qczy_button_permission` VALUES (111, '测试评估审批', 57, '一键测试', 'thirdparty:approve:test', 4);
INSERT INTO `qczy_button_permission` VALUES (112, '测试评估申请', 56, '模型申请', 'thirdparty:mul:modelApply', 6);
INSERT INTO `qczy_button_permission` VALUES (113, '测试评估任务', 58, '查看日志', 'thirdparty:assess:log', 1);
INSERT INTO `qczy_button_permission` VALUES (114, '测试评估任务', 58, '终止', 'thirdparty:assess:stop', 2);
INSERT INTO `qczy_button_permission` VALUES (115, '测试评估任务', 58, '继续', 'thirdparty:assess:continue', 3);
INSERT INTO `qczy_button_permission` VALUES (116, '测试评估任务', 58, '生成评估报告', 'thirdparty:assess:generate', 4);
INSERT INTO `qczy_button_permission` VALUES (117, '测试评估任务', 58, '查看评估报告', 'thirdparty:assess:report', 5);
INSERT INTO `qczy_button_permission` VALUES (118, '测试评估任务', 58, '重新开始', 'thirdparty:assess:restart', 6);
INSERT INTO `qczy_button_permission` VALUES (119, '测试评估任务', 58, '删除', 'thirdparty:assess:delete', 7);
INSERT INTO `qczy_button_permission` VALUES (120, '测试评估任务', 58, '开始评估', 'thirdparty:assess:start', 8);
INSERT INTO `qczy_button_permission` VALUES (121, '测试评估任务', 58, '创建评估任务', 'thirdparty:assess:createTask', 9);
INSERT INTO `qczy_button_permission` VALUES (122, '测试评估任务', 58, '对接厂商', 'thirdparty:assess:contact', 10);
INSERT INTO `qczy_button_permission` VALUES (123, '测试评估申请', 56, '删除', 'thirdparty:mul:delete', 7);
INSERT INTO `qczy_button_permission` VALUES (124, '测试评估审批', 57, '删除', 'thirdparty:approve:delete', 5);
INSERT INTO `qczy_button_permission` VALUES (125, '测试评估报告', 59, '导出任务评估报告', 'thirdparty:report:exportTask', 1);
INSERT INTO `qczy_button_permission` VALUES (126, '测试评估报告', 59, '导出申请号评估报告', 'thirdparty:report:exportApply', 2);
INSERT INTO `qczy_button_permission` VALUES (127, '测试评估任务', 58, '暂停', 'thirdparty:assess:pause', 11);
INSERT INTO `qczy_button_permission` VALUES (128, '测试评估申请', 56, '编辑', 'thirdparty:mul:edit', 8);
INSERT INTO `qczy_button_permission` VALUES (129, '测试评估任务', 58, '下载压缩包', 'thirdparty:assess:download', 12);
INSERT INTO `qczy_button_permission` VALUES (130, '测试评估报告', 59, '评估结果', 'thirdparty:report:result', 3);
INSERT INTO `qczy_button_permission` VALUES (131, '测试评估任务', 58, '编辑', 'thirdparty:assess:edit', 13);

-- ----------------------------
-- Table structure for qczy_computer_info
-- ----------------------------
DROP TABLE IF EXISTS `qczy_computer_info`;
CREATE TABLE `qczy_computer_info`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `cpu` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'cpu使用率',
  `gpu` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'gpu使用率',
  `create_date` date NULL DEFAULT NULL COMMENT '创建时间-年月日',
  `create_time` time NULL DEFAULT NULL COMMENT '时分秒',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_computer_info
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_data_father
-- ----------------------------
DROP TABLE IF EXISTS `qczy_data_father`;
CREATE TABLE `qczy_data_father`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `group_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '100001' COMMENT '数据集组id',
  `group_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '数据集组名称',
  `user_id` int(11) NOT NULL COMMENT '用户id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `data_type_id` int(11) NULL DEFAULT NULL,
  `is_third_party` int(8) NULL DEFAULT NULL COMMENT '是否为第三方数据-> 0：不是 1、是',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '数据集组表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_data_father
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_data_import_log
-- ----------------------------
DROP TABLE IF EXISTS `qczy_data_import_log`;
CREATE TABLE `qczy_data_import_log`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `son_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '数据集id',
  `file_size` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '文件大小',
  `file_ids` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '文件id',
  `user_id` int(11) NOT NULL COMMENT '用户id',
  `import_start_time` datetime NULL DEFAULT NULL COMMENT '导入开始时间',
  `import_end_time` datetime NULL DEFAULT NULL COMMENT '导入结束时间',
  `status` int(11) NOT NULL COMMENT '导入状态',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '数据集导入记录表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_data_import_log
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_data_son
-- ----------------------------
DROP TABLE IF EXISTS `qczy_data_son`;
CREATE TABLE `qczy_data_son`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `father_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '父id',
  `son_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '数据集id',
  `out_son_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '输出数据集id',
  `version` int(11) NOT NULL COMMENT '版本',
  `ano_type` int(11) NULL DEFAULT NULL COMMENT '标注类型 ->  0 图像分割 1 物体分割',
  `is_socket` int(11) NULL DEFAULT 0 COMMENT '走进度',
  `status` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标注状态（进度）',
  `remark` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '备注',
  `user_id` int(11) NOT NULL COMMENT '用户id',
  `file_ids` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '文件id',
  `create_time` datetime NOT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `is_many` int(8) NOT NULL DEFAULT 0 COMMENT '0->正常，1->多人标注',
  `is_third_party` int(8) NULL DEFAULT NULL COMMENT '是否为第三方数据集 0->不是 、1->是',
  `tag_selection_mode` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '判断选中的是  标签组 还是 标签',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '数据集表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_data_son
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_data_son_label
-- ----------------------------
DROP TABLE IF EXISTS `qczy_data_son_label`;
CREATE TABLE `qczy_data_son_label`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `son_id` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '数据集id',
  `label_id` int(11) NULL DEFAULT NULL COMMENT '标签id',
  `label_count` int(11) NULL DEFAULT 0 COMMENT '数据量',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_data_son_label
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_dept
-- ----------------------------
DROP TABLE IF EXISTS `qczy_dept`;
CREATE TABLE `qczy_dept`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `dept_name` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '部门名称',
  `sort` int(11) NULL DEFAULT NULL COMMENT '排序',
  `supt` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '负责人',
  `tele_phone` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '联系电话',
  `email` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '邮箱',
  `status` int(11) NULL DEFAULT NULL COMMENT '状态: 1-> 正常、 2->停用',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `is_allow_deletion` int(11) NOT NULL COMMENT '是否允许删除：1-> 不允许 、 2->允许',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '部门表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_dept
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_dict_data
-- ----------------------------
DROP TABLE IF EXISTS `qczy_dict_data`;
CREATE TABLE `qczy_dict_data`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '主键',
  `type_id` int(11) NOT NULL COMMENT '字典类型id',
  `parent_id` int(11) NOT NULL COMMENT '父id',
  `dict_sort` int(11) NULL DEFAULT NULL COMMENT '排序',
  `dict_label` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '字典标签',
  `dict_value` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '字典键值',
  `status` int(11) NULL DEFAULT NULL COMMENT '状态（1正常 2停用）',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `remark` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 125 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '字典-数据表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_dict_data
-- ----------------------------
INSERT INTO `qczy_dict_data` VALUES (17, 6, 1, 2, '图像分割', '2', 1, '2024-08-08 11:01:27', '2024-08-08 11:01:32', '数据集管理-（图片）标注类型');
INSERT INTO `qczy_dict_data` VALUES (18, 6, 14, 1, '文本分类', '1', 1, '2024-08-08 11:02:08', '2024-08-08 11:02:11', '数据集管理-（文本）标注类型');
INSERT INTO `qczy_dict_data` VALUES (19, 6, 14, 2, '序列标注', '2', 1, '2024-08-08 11:03:35', '2024-08-08 11:03:38', '数据集管理-（文本）标注类型');
INSERT INTO `qczy_dict_data` VALUES (20, 6, 15, 1, '视频分类', '1', 1, '2024-08-08 11:05:03', '2024-08-08 11:05:06', '数据集管理-（视频）标注类型');
INSERT INTO `qczy_dict_data` VALUES (21, 6, 15, 2, '视频检测', '2', 1, '2024-08-08 11:05:35', '2024-08-08 11:05:38', '数据集管理-（视频）标注类型');
INSERT INTO `qczy_dict_data` VALUES (22, 6, 16, 1, '矩形框标注', '1', 1, '2024-08-08 11:08:04', '2024-08-08 11:08:06', '数据集管理-（图片-物体检测）标注模板');
INSERT INTO `qczy_dict_data` VALUES (23, 6, 16, 2, '自定义四边形标注', '2', 1, '2024-08-08 11:08:59', '2024-08-08 11:09:02', '数据集管理-（图片-物体检测）标注模板');
INSERT INTO `qczy_dict_data` VALUES (24, 6, 17, 1, '实例分割', '1', 1, '2024-08-08 11:10:00', '2024-08-08 11:10:02', '数据集管理-（图片-图像分割）标注模板');
INSERT INTO `qczy_dict_data` VALUES (25, 6, 17, 2, '语义分割', '2', 1, '2024-08-08 11:11:25', '2024-08-08 11:11:27', '数据集管理-（图片-图像分割）标注模板');
INSERT INTO `qczy_dict_data` VALUES (26, 6, 18, 1, '短文本单标签', '1', 1, '2024-08-08 11:13:42', '2024-08-08 11:13:45', '数据集管理-（文本-文本分类）标注模板');
INSERT INTO `qczy_dict_data` VALUES (27, 6, 18, 2, '短文本多标签', '2', 1, '2024-08-08 11:14:27', '2024-08-08 11:14:30', '数据集管理-（文本-文本分类）标注模板');
INSERT INTO `qczy_dict_data` VALUES (28, 6, 19, 1, 'IOB标注模式', '1', 1, '2024-08-08 11:15:24', '2024-08-08 11:15:27', '数据集管理-（文本-序列标注）标注模板');
INSERT INTO `qczy_dict_data` VALUES (29, 6, 19, 2, 'IO标注模式', '2', 1, '2024-08-08 11:16:38', '2024-08-08 11:16:41', '数据集管理-（文本-序列标注）标注模板');
INSERT INTO `qczy_dict_data` VALUES (30, 6, 20, 1, '短视频单标签', '1', 1, '2024-08-08 11:17:43', '2024-08-08 11:17:46', '数据集管理-（视频-视频分类）标注模板');
INSERT INTO `qczy_dict_data` VALUES (31, 6, 21, 1, '视频检测', '1', 1, '2024-08-08 11:18:43', '2024-08-08 11:18:47', '数据集管理-（视频-视频检测）标注模板');
INSERT INTO `qczy_dict_data` VALUES (32, 15, 0, 1, '1', '1', 1, '2024-08-08 14:29:44', NULL, '1');
INSERT INTO `qczy_dict_data` VALUES (36, 15, 33, 2, '2323', '2323', 1, '2024-08-08 14:47:21', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (37, 7, 0, 1, '图像类', '1', 1, '2024-09-03 15:48:33', '2024-09-03 15:53:28', '数据类型');
INSERT INTO `qczy_dict_data` VALUES (38, 7, 37, 1, '图像分类', '1', 1, '2024-09-03 15:53:57', NULL, '标注类型');
INSERT INTO `qczy_dict_data` VALUES (39, 7, 37, 2, '物体检测', '2', 1, '2024-09-03 15:54:23', NULL, '标注类型');
INSERT INTO `qczy_dict_data` VALUES (40, 7, 37, 3, '图像分割', '3', 1, '2024-09-03 15:54:56', NULL, '标注类型');
INSERT INTO `qczy_dict_data` VALUES (41, 7, 39, 1, '图片全局增强', '1', 1, '2024-09-03 15:55:39', NULL, ' 增强区域');
INSERT INTO `qczy_dict_data` VALUES (42, 7, 39, 2, '全局以及标注框局部增强', '2', 1, '2024-09-03 15:56:10', NULL, '增强区域');
INSERT INTO `qczy_dict_data` VALUES (43, 7, 40, 1, '图片全局增强', '1', 1, '2024-09-03 15:57:21', NULL, '增强区域');
INSERT INTO `qczy_dict_data` VALUES (48, 8, 0, 1, '变电', '变电', 1, '2024-09-25 16:37:36', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (50, 8, 48, 1, '输电', '输电', 1, '2024-09-25 16:38:52', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (51, 8, 48, 1, '导线', '导线', 1, '2024-09-25 16:39:21', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (53, 6, 0, 7, '输电', '7', 1, '2024-09-25 16:44:14', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (54, 6, 53, 1, '驱鸟器', '1', 1, '2024-09-25 16:44:28', '2024-10-15 11:04:05', '');
INSERT INTO `qczy_dict_data` VALUES (55, 6, 53, 2, '绝缘子', '2', 1, '2024-09-25 16:44:49', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (56, 9, 0, 1, '001', '', NULL, '2024-09-26 12:26:03', NULL, '1');
INSERT INTO `qczy_dict_data` VALUES (57, 9, 0, 1, '002', '', NULL, '2024-09-26 12:26:10', NULL, '1');
INSERT INTO `qczy_dict_data` VALUES (58, 9, 56, 1, '0001', '', NULL, '2024-09-26 12:26:16', NULL, '1');
INSERT INTO `qczy_dict_data` VALUES (59, 9, 0, 3, '003', '', NULL, '2024-09-26 12:26:23', NULL, '3');
INSERT INTO `qczy_dict_data` VALUES (61, 8, 0, 1, '输电', '', NULL, '2024-09-30 08:38:46', NULL, '输电');
INSERT INTO `qczy_dict_data` VALUES (62, 8, 61, 1, '绝缘子', '', NULL, '2024-09-30 08:39:03', NULL, '绝缘子');
INSERT INTO `qczy_dict_data` VALUES (63, 8, 61, 2, '导地线', '', NULL, '2024-09-30 08:39:16', NULL, '导地线');
INSERT INTO `qczy_dict_data` VALUES (64, 8, 61, 3, '杆塔', '', NULL, '2024-09-30 08:39:31', NULL, '杆塔');
INSERT INTO `qczy_dict_data` VALUES (65, 6, 0, 12, '变电', '', NULL, '2024-09-30 08:39:54', NULL, '变电');
INSERT INTO `qczy_dict_data` VALUES (66, 6, 65, 2, '杆塔', '', NULL, '2024-09-30 08:40:04', NULL, '杆塔');
INSERT INTO `qczy_dict_data` VALUES (68, 6, 53, 8, '杆塔', '', NULL, '2024-09-30 08:41:38', NULL, '杆塔');
INSERT INTO `qczy_dict_data` VALUES (69, 10, 0, 1, 'land', '', NULL, '2024-10-12 12:53:27', '2024-10-12 15:22:28', '地面');
INSERT INTO `qczy_dict_data` VALUES (70, 10, 0, 2, 'tower', '', NULL, '2024-10-12 12:58:15', NULL, '杆塔');
INSERT INTO `qczy_dict_data` VALUES (71, 10, 0, 3, 'sky', '', NULL, '2024-10-12 12:58:28', NULL, '天空');
INSERT INTO `qczy_dict_data` VALUES (72, 11, 0, 1, 'tower crane', '', NULL, '2024-10-12 12:59:16', NULL, '塔吊');
INSERT INTO `qczy_dict_data` VALUES (73, 11, 0, 2, 'crane at work', '', NULL, '2024-10-12 12:59:43', NULL, '吊车工作');
INSERT INTO `qczy_dict_data` VALUES (74, 11, 0, 3, 'crane not at work', '', NULL, '2024-10-12 12:59:58', NULL, '吊车不工作');
INSERT INTO `qczy_dict_data` VALUES (75, 11, 0, 4, 'Cement pump truck at work', '', NULL, '2024-10-12 13:00:10', NULL, '水泥泵车工作');
INSERT INTO `qczy_dict_data` VALUES (76, 11, 0, 5, 'Cement pump truck not at work', '', NULL, '2024-10-12 13:00:24', NULL, '水泥泵车不工作');
INSERT INTO `qczy_dict_data` VALUES (77, 11, 0, 6, 'pile driver', '', NULL, '2024-10-12 13:00:38', NULL, '打桩机');
INSERT INTO `qczy_dict_data` VALUES (78, 11, 0, 7, 'cement mixer', '', NULL, '2024-10-12 13:00:51', NULL, '水泥搅拌车');
INSERT INTO `qczy_dict_data` VALUES (79, 11, 0, 8, 'bulldozer', '', NULL, '2024-10-12 13:01:02', NULL, '推土机');
INSERT INTO `qczy_dict_data` VALUES (80, 11, 0, 9, 'excavating machinery', '', NULL, '2024-10-12 13:01:17', NULL, '挖掘机');
INSERT INTO `qczy_dict_data` VALUES (82, 10, 69, 11, 'tower crane', '', NULL, '2024-10-12 13:10:37', '2024-10-12 15:35:26', '塔吊');
INSERT INTO `qczy_dict_data` VALUES (83, 10, 69, 2, 'crane at work', '', NULL, '2024-10-12 13:10:47', NULL, '吊车工作(crane at work)');
INSERT INTO `qczy_dict_data` VALUES (84, 10, 69, 3, 'crane not at work', '', NULL, '2024-10-12 13:10:58', NULL, '吊车不工作(crane not at work)');
INSERT INTO `qczy_dict_data` VALUES (85, 10, 69, 4, 'Cement pump truck at work', '', NULL, '2024-10-12 13:11:10', NULL, '水泥泵车工作(Cement pump truck at work)');
INSERT INTO `qczy_dict_data` VALUES (86, 10, 69, 5, 'Cement pump truck not at work', '', NULL, '2024-10-12 13:11:21', NULL, '水泥泵车不工作(Cement pump truck not at work)');
INSERT INTO `qczy_dict_data` VALUES (88, 10, 69, 6, 'pile driver', '', NULL, '2024-10-12 13:11:54', NULL, '打桩机(pile driver)');
INSERT INTO `qczy_dict_data` VALUES (89, 10, 69, 7, 'cement mixer', '', NULL, '2024-10-12 13:12:05', NULL, '水泥搅拌车(cement mixer)');
INSERT INTO `qczy_dict_data` VALUES (90, 10, 69, 8, 'bulldozer', '', NULL, '2024-10-12 13:12:18', NULL, '推土机(bulldozer)');
INSERT INTO `qczy_dict_data` VALUES (91, 10, 69, 9, 'excavating machinery', '', NULL, '2024-10-12 13:12:30', NULL, '挖掘机(excavating machinery)');
INSERT INTO `qczy_dict_data` VALUES (92, 10, 70, 1, 'bird nest', '', NULL, '2024-10-12 13:12:50', NULL, '鸟窝(bird nest)');
INSERT INTO `qczy_dict_data` VALUES (94, 10, 71, 2, 'sunshade net', '', NULL, '2024-10-12 13:13:43', NULL, '遮阳网(sunshade net)');
INSERT INTO `qczy_dict_data` VALUES (96, 10, 71, 3, 'Hanging balloons online', '', NULL, '2024-10-12 13:15:05', NULL, '气球（Hanging balloons online/balloons）');
INSERT INTO `qczy_dict_data` VALUES (98, 10, 71, 1, 'Plastic film', '', NULL, '2024-10-12 13:16:34', NULL, '塑料薄膜(Plastic film)');
INSERT INTO `qczy_dict_data` VALUES (99, 10, 71, 4, 'A kite hanging online', '', NULL, '2024-10-12 13:16:54', '2024-10-16 10:35:02', '风筝（A kite hanging online/kite）');
INSERT INTO `qczy_dict_data` VALUES (100, 12, 0, 1, 'insulator', '', NULL, '2024-10-14 10:17:20', NULL, '绝缘子');
INSERT INTO `qczy_dict_data` VALUES (101, 12, 0, 2, 'damper', '', NULL, '2024-10-14 10:17:32', NULL, '防振锤');
INSERT INTO `qczy_dict_data` VALUES (102, 12, 0, 3, 'windbirdrepellent', '', NULL, '2024-10-14 10:17:52', NULL, '防鸟设施');
INSERT INTO `qczy_dict_data` VALUES (103, 6, 53, 4, '防震锤', '', NULL, '2024-10-15 11:04:19', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (104, 6, 65, 1, '绝缘子', '', NULL, '2024-10-15 11:04:31', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (106, 6, 105, 2, 'child1', '', NULL, '2024-10-16 10:00:58', NULL, '1');
INSERT INTO `qczy_dict_data` VALUES (109, 6, 108, 1, '001', '', NULL, '2024-10-17 12:50:36', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (110, 6, 109, 1, '0001', '', NULL, '2024-10-17 12:50:47', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (111, 6, 108, 2, '002', '', NULL, '2024-10-17 12:50:55', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (112, 13, 0, 1, '正常', '', NULL, '2024-10-17 13:57:30', '2024-10-22 16:57:26', '0');
INSERT INTO `qczy_dict_data` VALUES (113, 13, 0, 2, '异常', '', NULL, '2024-10-17 13:57:37', '2024-10-22 16:57:33', '1');
INSERT INTO `qczy_dict_data` VALUES (122, 14, 0, 1, '绝缘子处理模型', '', NULL, '2024-10-22 17:48:10', '2024-11-01 20:57:54', '/app/runs/train/ghf');
INSERT INTO `qczy_dict_data` VALUES (123, 6, 0, 99, '第三方数据集', '', NULL, '2025-05-21 10:47:27', NULL, '');
INSERT INTO `qczy_dict_data` VALUES (124, 6, 123, 1, '杆塔', '', NULL, '2025-05-21 10:55:58', NULL, '');

-- ----------------------------
-- Table structure for qczy_dict_type
-- ----------------------------
DROP TABLE IF EXISTS `qczy_dict_type`;
CREATE TABLE `qczy_dict_type`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '字典主键',
  `dict_name` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '字典名称',
  `status` int(11) NULL DEFAULT 0 COMMENT '状态（1正常 2停用）',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `remark` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '备注',
  `is_allow_deletion` int(11) NULL DEFAULT NULL COMMENT '是否允许删除：1-> 不允许 、 2->允许',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 15 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '字典-类型表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_dict_type
-- ----------------------------
INSERT INTO `qczy_dict_type` VALUES (6, '数据集类型', 1, '2024-08-09 10:16:39', '2024-09-05 21:16:06', NULL, 1);
INSERT INTO `qczy_dict_type` VALUES (7, '任务类型', 1, '2024-09-03 15:48:08', '2024-09-03 15:51:01', '数据增强-任务类型', 1);
INSERT INTO `qczy_dict_type` VALUES (10, '区域生成选择框', 0, '2024-10-12 12:41:36', '2024-10-12 13:10:17', '区域生成下拉枚举', 1);
INSERT INTO `qczy_dict_type` VALUES (12, '缺陷生成选择框', 0, '2024-10-14 10:16:25', NULL, '', 1);
INSERT INTO `qczy_dict_type` VALUES (14, '自动标注-模型列表', 0, '2024-10-17 15:48:40', '2024-10-22 17:47:41', '自动标注-模型列表', NULL);

-- ----------------------------
-- Table structure for qczy_file
-- ----------------------------
DROP TABLE IF EXISTS `qczy_file`;
CREATE TABLE `qczy_file`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `fd_name` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件名称',
  `fd_type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件类型：(jpg、png、txt...)',
  `fd_suffix` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件后缀',
  `fd_path` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件正式路径',
  `fd_access_path` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件访问路径',
  `http_file_path` varchar(1000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '前端文件访问路径',
  `fd_size` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件大小',
  `width` int(11) NULL DEFAULT NULL COMMENT '宽',
  `height` int(11) NULL DEFAULT NULL COMMENT '高',
  `operate_width` int(11) NULL DEFAULT NULL,
  `operate_height` int(11) NULL DEFAULT NULL,
  `file_status` int(11) NULL DEFAULT NULL COMMENT '文件状态->0:原始图片、1:训练结果图片',
  `task_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务批次->哪次任务生成的文件',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_qczy_file_status`(`file_status`) USING BTREE,
  INDEX `idx_qczy_file_id_status`(`id`, `file_status`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '正式文件表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_file
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_label
-- ----------------------------
DROP TABLE IF EXISTS `qczy_label`;
CREATE TABLE `qczy_label`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `only_id` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '唯一id',
  `label_group_id` int(11) NOT NULL COMMENT '标签组id',
  `label_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标签名称',
  `english_label_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '英文标签名称',
  `label_sort` int(8) NULL DEFAULT NULL COMMENT '标签排序',
  `label_color` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标签颜色',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '标签表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_label
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_label_group
-- ----------------------------
DROP TABLE IF EXISTS `qczy_label_group`;
CREATE TABLE `qczy_label_group`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `label_group_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标签组名称',
  `english_label_group_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标签组英文名称',
  `label_group_desc` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '标签组描述',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '标签组表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_label_group
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_login_log
-- ----------------------------
DROP TABLE IF EXISTS `qczy_login_log`;
CREATE TABLE `qczy_login_log`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '访问ID',
  `login_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '登录账号',
  `ipaddr` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '登录IP地址',
  `browser` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '浏览器类型',
  `os` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '操作系统',
  `login_time` datetime NOT NULL COMMENT '访问时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_sys_logininfor_lt`(`login_time`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '系统访问记录' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_login_log
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_many_assign
-- ----------------------------
DROP TABLE IF EXISTS `qczy_many_assign`;
CREATE TABLE `qczy_many_assign`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `many_mark_id` int(8) NULL DEFAULT NULL COMMENT '多人标注任务id',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `user_id` int(8) NULL DEFAULT NULL COMMENT '分配用户id',
  `assign_file_ids` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '分配文件id集合',
  `yes_mark` int(8) NULL DEFAULT NULL COMMENT '已完成标注数量',
  `no_mark` int(8) NULL DEFAULT NULL COMMENT '未完成标注数量',
  `progress` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '完成进度',
  `user_state` int(8) NULL DEFAULT NULL COMMENT '用户状态',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `is_lose` int(8) NULL DEFAULT NULL COMMENT '此数据是否失效 0：正常 、 1：失效',
  `is_submit` int(8) NULL DEFAULT NULL COMMENT '是否提交 0：未提交 、 1：已提交',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '多人任务分配表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_many_assign
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_many_audit
-- ----------------------------
DROP TABLE IF EXISTS `qczy_many_audit`;
CREATE TABLE `qczy_many_audit`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `many_mark_id` int(8) NULL DEFAULT NULL COMMENT '任务id',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `user_id` int(8) NULL DEFAULT NULL COMMENT '审核人员id',
  `mark_user_id` int(8) NULL DEFAULT NULL COMMENT '标注人员id',
  `audit_file_ids` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '审核文件集',
  `audit_state` int(8) NULL DEFAULT NULL COMMENT '审核状态',
  `yes_examine` int(8) NULL DEFAULT NULL COMMENT '已审核',
  `no_examine` int(8) NULL DEFAULT NULL COMMENT '未审核',
  `progress` int(8) NULL DEFAULT NULL COMMENT '审核进度',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `is_lose` int(8) NULL DEFAULT NULL COMMENT '此数据是否失效 0：正常 、 1：失效',
  `is_submit` int(8) NULL DEFAULT NULL COMMENT '是否提交 0：未提交 、 1：已提交',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_many_audit
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_many_file
-- ----------------------------
DROP TABLE IF EXISTS `qczy_many_file`;
CREATE TABLE `qczy_many_file`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `task_id` int(11) NULL DEFAULT NULL COMMENT '任务id',
  `user_id` int(11) NULL DEFAULT NULL COMMENT '标注员id',
  `file_id` int(8) NULL DEFAULT NULL COMMENT '文件id',
  `audit_user_id` int(8) NULL DEFAULT NULL COMMENT '审核员id',
  `is_approve` int(8) NULL DEFAULT NULL COMMENT '是否通过-> 0: 未审核 、 1：已通过 、 2：未通过',
  `not_pass_message` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '不通过验收意见',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_taskId_auditUserId`(`task_id`, `audit_user_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_many_file
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_many_mark
-- ----------------------------
DROP TABLE IF EXISTS `qczy_many_mark`;
CREATE TABLE `qczy_many_mark`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `task_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务名称',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `team_id` int(8) NULL DEFAULT NULL COMMENT '团队id',
  `audit_team_id` int(8) NULL DEFAULT NULL COMMENT '审核团队id',
  `task_state` int(8) NULL DEFAULT NULL COMMENT '任务状态',
  `user_id` int(8) NULL DEFAULT NULL COMMENT '创建用户id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '多人标注任务表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_many_mark
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_mark_info
-- ----------------------------
DROP TABLE IF EXISTS `qczy_mark_info`;
CREATE TABLE `qczy_mark_info`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '数据集id',
  `file_id` int(11) NOT NULL COMMENT '文件id',
  `mark_file_id` int(11) NULL DEFAULT NULL COMMENT '标注文件id',
  `labels` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '标签值',
  `mark_info` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '标注信息（json格式）',
  `label_mark_info` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT 'web标注信息（json格式）',
  `width` int(11) NULL DEFAULT NULL COMMENT '长',
  `operate_width` int(11) NULL DEFAULT NULL COMMENT '标记长',
  `operate_height` int(11) NULL DEFAULT NULL COMMENT '标记宽',
  `height` int(11) NULL DEFAULT NULL COMMENT '宽',
  `is_invalid` int(11) NULL DEFAULT 1 COMMENT '是否为无效数据->(0：无效、1：有效)',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_qczy_mark_info_file_id`(`file_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '数据标注（点位信息）表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_mark_info
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_menu
-- ----------------------------
DROP TABLE IF EXISTS `qczy_menu`;
CREATE TABLE `qczy_menu`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `parent_id` int(11) NOT NULL COMMENT '父级菜单id ',
  `menu_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL DEFAULT '' COMMENT '菜单名称',
  `icon` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '菜单图标',
  `local_icon` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `permissions` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `i18nKey` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `web_path` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'web页面路径',
  `component` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '对应路由里面的组件component',
  `active_menu` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `type` int(11) NOT NULL COMMENT '权限类型：0->目录；1->菜单；2->按钮（接口绑定权限）',
  `sort` int(11) NOT NULL COMMENT '排序',
  `create_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  `hide_in_menu` int(1) NOT NULL COMMENT '是否显示  0：不显示 1：显示',
  `is_deleted` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否删除    0：未删除，1：已删除',
  `href` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 66 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '菜单表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_menu
-- ----------------------------
INSERT INTO `qczy_menu` VALUES (1, 0, 'home', 'mdi:monitor-dashboard', NULL, NULL, 'route.home', '/home', 'layout.base$view.home', NULL, 1, 1, '2025-03-06 14:10:22', '2025-03-06 14:10:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (2, 0, 'data-manage', 'bxs:data', NULL, NULL, 'route.data-manage', '/data-manage', 'layout.base', NULL, 0, 2, '2025-03-06 14:10:22', '2025-03-06 14:10:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (3, 2, 'data-manage_maplist', 'material-symbols-light:dataset-sharp', NULL, '', 'route.data-manage_maplist', '/data-manage/maplist', 'view.data-manage_maplist', NULL, 1, 1, '2025-03-06 15:29:03', '2025-03-06 15:29:03', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (4, 3, 'data-manage_map', 'material-symbols-light:dataset-sharp', NULL, NULL, 'route.data-manage_map', '/data-manage/map', 'view.data-manage_map', 'data-manage_maplist', 2, 1, '2025-03-07 11:35:20', '2025-03-07 11:35:20', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (5, 3, 'data-ano_detail', NULL, NULL, NULL, 'route.data-ano_detail', '/data-ano/detail', 'view.data-ano_detail', 'data-manage_maplist', 2, 2, '2025-03-07 11:35:20', '2025-03-07 11:35:20', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (6, 3, 'data-ano_operation', NULL, NULL, NULL, 'route.data-ano_operation', '/data-ano/operation', 'view.data-ano_operation', 'data-manage_maplist', 2, 3, '2025-03-07 11:35:20', '2025-03-07 11:35:20', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (7, 3, 'dataset_operate', NULL, NULL, NULL, 'route.dataset_operate', '/dataset/operate', 'view.dataset_operate', 'data-manage_maplist', 2, 4, '2025-03-07 11:35:20', '2025-03-07 11:35:20', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (8, 3, 'data-manage_detail', NULL, NULL, NULL, 'route.data-manage_detail', '/data-manage/detail', 'view.data-manage_detail', 'data-manage_maplist', 2, 5, '2025-03-07 11:35:20', '2025-03-07 11:35:20', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (9, 0, 'data-ano', 'pajamas:labels', NULL, NULL, 'route.data-ano', '/data-ano', 'layout.base', NULL, 0, 3, '2025-03-06 14:10:22', '2025-03-06 14:10:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (10, 9, 'data-ano_group', 'material-symbols-light:ad-group', NULL, NULL, 'route.data-ano_group', '/data-ano/group', 'view.data-ano_group', NULL, 1, 1, '2025-03-06 14:10:22', '2025-03-06 14:10:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (11, 10, 'data-ano_gtag', NULL, NULL, NULL, 'route.data-ano_gtag', '/data-ano/gtag', 'view.data-ano_gtag', 'data-ano_group', 2, 1, '2025-03-07 11:35:39', '2025-03-07 11:35:39', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (12, 9, 'data-ano_online', 'pajamas:label', NULL, NULL, 'route.data-ano_online', '/data-ano/online', 'view.data-ano_online', NULL, 1, 2, '2025-03-06 14:10:22', '2025-03-06 14:10:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (13, 9, 'data-ano_autoano', '', 'ic--baseline-auto-graph', NULL, 'route.data-ano_autoano', '/data-ano/autoano', 'view.data-ano_autoano', NULL, 1, 3, '2025-03-06 14:13:37', '2025-03-06 14:13:37', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (14, 13, 'data-expansion_exportres', NULL, NULL, NULL, 'route.data-expansion_exportres', '/data-expansion/exportres', 'view.data-expansion_exportres', NULL, 2, 1, '2025-03-07 11:35:42', '2025-03-07 11:35:42', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (15, 13, 'data-expansion_addmap', NULL, NULL, NULL, 'route.data-expansion_addmap', '/data-expansion/addmap', 'view.data-expansion_addmap', NULL, 2, 2, '2025-03-07 11:35:42', '2025-03-07 11:35:42', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (16, 9, 'data-ano_mulano', '', 'data-ano_mulano', '', 'route.data-ano_mulano', '/data-ano/mulano', 'view.data-ano_mulano', NULL, 1, 3, '2025-03-06 15:29:06', '2025-03-06 15:29:06', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (17, 16, 'data-ano_groupmanage', NULL, NULL, '', 'route.data-ano_groupmanage', '/data-ano/groupmanage', 'view.data-ano_groupmanage', 'data-ano_mulano', 2, 1, '2025-03-06 14:47:28', '2025-03-06 14:47:28', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (18, 16, 'data-ano_mulanotask', '', NULL, NULL, 'route.data-ano_mulanotask', '/data-ano/mulanotask', 'view.data-ano_mulanotask', 'data-ano_mulano', 2, 2, '2025-03-06 14:47:28', '2025-03-06 14:47:28', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (19, 0, 'data-expansion', 'material-symbols:folder-data', NULL, NULL, 'route.data-expansion', '/data-expansion', 'layout.base', NULL, 0, 4, '2025-03-06 14:10:30', '2025-03-06 14:10:30', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (20, 19, 'data-expansion_errarea', 'icon-park-outline:area-map', NULL, NULL, 'route.data-expansion_errarea', '/data-expansion/errarea', 'view.data-expansion_errarea', NULL, 1, 1, '2025-03-06 14:10:30', '2025-03-06 14:10:30', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (21, 20, 'data-expansion_exportres', NULL, NULL, NULL, 'route.data-expansion_exportres', '/data-expansion/exportres', 'view.data-expansion_exportres', NULL, 2, 1, '2025-03-07 11:35:47', '2025-03-07 11:35:47', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (22, 20, 'data-expansion_add', NULL, NULL, '', 'route.data-expansion_add', '/data-expansion/add', 'view.data-expansion_add', NULL, 2, 2, '2025-03-07 11:35:47', '2025-03-07 11:35:47', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (23, 20, 'data-expansion_addmap', NULL, NULL, NULL, 'route.data-expansion_addmap', '/data-expansion/addmap', 'view.data-expansion_addmap', NULL, 2, 3, '2025-03-07 11:35:47', '2025-03-07 11:35:47', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (24, 19, 'data-expansion_scenechange', 'icon-park:change', NULL, NULL, 'route.data-expansion_scenechange', '/data-expansion/scenechange', 'view.data-expansion_scenechange', NULL, 1, 2, '2025-03-06 14:10:30', '2025-03-06 14:10:30', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (25, 24, 'data-expansion_add', NULL, NULL, NULL, 'route.data-expansion_add', '/data-expansion/add', 'view.data-expansion_add', NULL, 2, 1, '2025-03-07 11:35:52', '2025-03-07 11:35:52', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (26, 19, 'data-expansion_imgopmaster', '', 'ImgOpMaster', NULL, 'route.data-expansion_imgopmaster', '/data-expansion/imgopmaster', 'view.data-expansion_imgopmaster', NULL, 1, 3, '2025-03-06 14:10:30', '2025-03-06 14:10:30', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (27, 26, 'data-expansion_imgtask', NULL, NULL, NULL, 'route.data-expansion_imgtask', '/data-expansion/imgtask', 'view.data-expansion_imgtask', 'data-expansion_imgopmaster', 2, 1, '2025-03-06 14:14:04', '2025-03-06 14:14:04', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (28, 26, 'data-expansion_exportres', NULL, NULL, NULL, 'route.data-expansion_exportres', '/data-expansion/exportres', 'view.data-expansion_exportres', NULL, 2, 2, '2025-03-06 14:14:04', '2025-03-06 14:14:04', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (29, 0, 'model-manage', 'carbon:model', NULL, NULL, 'route.model-manage', '/model-manage', 'layout.base', NULL, 0, 5, '2025-03-06 14:14:07', '2025-03-06 14:14:07', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (30, 29, 'model-manage_default', 'octicon:ai-model-16', NULL, NULL, 'route.model-manage_default', '/model-manage/default', 'view.model-manage_default', NULL, 1, 1, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (31, 29, 'model-manage_train', 'material-symbols:train-outline', NULL, NULL, 'route.model-manage_train', '/model-manage/train', 'view.model-manage_train', NULL, 1, 2, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (32, 31, 'model-manage_config', NULL, NULL, NULL, 'route.model-manage_config', '/model-manage/config', 'view.model-manage_config', 'model-manage_train', 2, 1, '2025-04-08 11:30:43', '2025-04-08 11:30:43', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (33, 29, 'model-manage_assess', '', 'ic--baseline-assessment', NULL, 'route.model-manage_assess', '/model-manage/assess', 'view.model-manage_assess', NULL, 1, 3, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (40, 0, 'manage', 'carbon:cloud-service-management', NULL, NULL, 'route.manage', '/manage', 'layout.base', NULL, 0, 8, '2025-06-30 10:05:01', '2025-06-30 10:05:01', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (41, 40, 'manage_dept', 'mingcute:department-fill', NULL, NULL, 'route.manage_dept', '/manage/dept', 'view.manage_dept', NULL, 1, 3, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (42, 40, 'manage_modelconfig', 'octicon:ai-model-16', NULL, NULL, 'route.manage_modelconfig', '/manage/modelconfig', 'view.manage_modelconfig', NULL, 1, 1, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (43, 40, 'manage_dict', 'material-symbols:dictionary', NULL, NULL, 'route.manage_dict', '/manage/dict', 'view.manage_dict', NULL, 1, 2, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (44, 43, 'manage_dict-type', 'route.manage_dict-type', NULL, NULL, NULL, '/manage/dict-type', 'view.manage_dict-type', NULL, 2, 1, '2025-03-06 14:14:43', '2025-03-06 14:14:43', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (45, 40, 'manage_user', 'mdi:user', NULL, NULL, 'route.manage_user', '/manage/user', 'view.manage_user', NULL, 1, 4, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (46, 40, 'manage_role', 'carbon:user-role', NULL, NULL, 'route.manage_role', '/manage/role', 'view.manage_role', NULL, 1, 5, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (47, 40, 'manage_log', 'pajamas:log', NULL, NULL, 'route.manage_log', '/manage/log', NULL, NULL, 0, 8, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (48, 47, 'manage_log_login', 'ic:round-login', NULL, NULL, 'route.manage_log_login', '/manage/log/login', 'view.manage_log_login', NULL, 1, 1, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (49, 47, 'manage_log_operate', 'tdesign:cooperate', NULL, NULL, 'route.manage_log_operate', '/manage/log/operate', 'view.manage_log_operate', NULL, 1, 2, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (50, 16, 'data-ano_operation', NULL, NULL, NULL, 'route.data-ano_operation', '/data-ano/operation', 'view.data-ano_operation', NULL, 2, 3, '2025-03-06 14:14:47', '2025-03-06 14:14:47', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (51, 0, 'login', '', NULL, NULL, 'route.login', '/login/:module(pwd-login|code-login|register|reset-pwd|bind-wechat)?', 'layout.blank$view.login', NULL, 1, 9, '2025-03-06 14:10:49', '2025-03-06 14:10:49', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (53, 16, 'data-ano_operation', NULL, NULL, NULL, 'route.data-ano_operation', '/data-ano/operation', 'view.data-ano_operation', 'data-manage_maplist', 2, 3, '2025-03-07 11:36:05', '2025-03-07 11:36:05', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (54, 9, 'data-ano_imgoperate', NULL, 'ep--operation', NULL, 'route.data-ano_imgoperate', '/data-ano/imgoperate', 'view.data-ano_imgoperate', NULL, 2, 5, '2025-04-08 11:32:33', '2025-04-08 11:32:33', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (55, 0, 'ThirdParty', NULL, 'ThirdParty', NULL, 'route.ThirdParty', '/thirdparty', 'layout.base', NULL, 0, 6, '2025-05-29 16:07:31', '2025-05-29 16:07:31', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (56, 55, 'thirdparty_modelmanage', NULL, 'ThirdParty_ModelManage', NULL, 'route.thirdparty_modelmanage', '/thirdparty/modelmanage', 'view.thirdparty_modelmanage', NULL, 1, 1, '2025-05-29 10:01:22', '2025-05-29 10:01:22', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (57, 55, 'thirdparty_approve', NULL, 'ThirdParty_Approve', NULL, 'route.thirdparty_approve', '/thirdparty/approve', 'view.thirdparty_approve', NULL, 1, 2, '2025-05-29 10:03:11', '2025-05-29 10:03:11', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (58, 55, 'ThirdParty_Assess', NULL, 'ThirdParty_Assess', NULL, 'route.ThirdParty_Assess', '/thirdparty/assess', 'view.thirdparty_assess', NULL, 1, 3, '2025-05-29 16:08:02', '2025-05-29 16:08:02', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (59, 55, 'thirdparty_report', NULL, 'ThirdParty_Report', NULL, 'route.thirdparty_report', '/thirdparty/report', 'view.thirdparty_report', NULL, 1, 4, '2025-05-29 10:06:37', '2025-05-29 10:06:37', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (60, 56, 'thirdparty_modeloperate', NULL, 'ThirdParty_ModelOperate', NULL, 'route.thirdparty_modeloperate', '/thirdparty/modeloperate', 'view.thirdparty_modeloperate', 'thirdparty_modelmanage', 2, 1, '2025-05-29 14:38:04', '2025-05-29 14:38:04', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (61, 58, 'thirdparty_createtask', NULL, 'ThirdParty_CreateTask', NULL, 'route.thirdparty_createtask', '/thirdparty/createtask', 'view.thirdparty_createtask', 'ThirdParty_Assess', 2, 1, '2025-06-05 14:19:19', '2025-06-05 14:19:19', 1, 0, NULL);
INSERT INTO `qczy_menu` VALUES (62, 0, 'boxpulse', NULL, 'boxpulse', NULL, 'route.boxpulse', '/boxpulse', 'layout.base', NULL, 0, 7, '2025-06-30 10:05:00', '2025-06-30 10:05:00', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (63, 62, 'boxpulse_code-test', NULL, 'boxpulse_code-test', NULL, 'route.boxpulse_code-test', '/boxpulse/code-test', 'view.boxpulse_code-test', NULL, 1, 2, '2025-07-08 11:43:06', '2025-07-08 11:43:06', 0, 0, 'http://121.41.225.75:7689/sonarqube/');
INSERT INTO `qczy_menu` VALUES (64, 62, 'boxpulse_test', NULL, 'boxpulse_test', NULL, 'route.boxpulse_test', '/boxpulse/test', 'view.boxpulse_test', NULL, 1, 1, '2025-07-01 10:31:03', '2025-07-01 10:31:03', 0, 0, NULL);
INSERT INTO `qczy_menu` VALUES (65, 16, 'data-ano_imgoperate', NULL, NULL, NULL, 'route.data-ano_imgoperate', '/data-ano/imgoperate', 'view.data-ano_imgoperate', 'data-manage_maplist', 2, 3, '2025-08-05 10:52:38', '2025-08-05 10:52:38', 0, 0, NULL);

-- ----------------------------
-- Table structure for qczy_model_assess_config
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_assess_config`;
CREATE TABLE `qczy_model_assess_config`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `assess_task_id` int(8) NULL DEFAULT NULL COMMENT '评估任务id',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `model_address` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型接口地址',
  `request_type` int(8) NULL DEFAULT NULL COMMENT '模型传输方式(请求类型)',
  `model_file_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型参数文件名',
  `model_params` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '模型参数',
  `assess_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '评估描述',
  `assess_target` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '评估指标',
  `assess_chart` varchar(300) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '评估图表',
  `error_message` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '错误信息',
  `label_map` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `assess_target_map` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `model_params_path` varchar(300) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `model_class` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '模型识别类型',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型评估配置表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_assess_config
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_assess_task
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_assess_task`;
CREATE TABLE `qczy_model_assess_task`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `task_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务名称',
  `task_type` int(8) NULL DEFAULT NULL COMMENT '任务类型 - > 1：测试 、2：评估',
  `task_status` int(8) NULL DEFAULT NULL COMMENT '任务状态 - > 1：待执行 、 2：执行中 、3：已完成 、 4：任务失败 、5：终止 、6：继续',
  `task_progress` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务进度',
  `model_base_id` int(8) NULL DEFAULT NULL COMMENT '模型基础信息id',
  `task_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务描述',
  `task_version` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '任务版本',
  `version_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '版本描述',
  `user_id` int(8) NULL DEFAULT NULL COMMENT '用户id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `task_result` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型评估任务表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_assess_task
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_base
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_base`;
CREATE TABLE `qczy_model_base`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `apply_for_num` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '申请单号',
  `model_name` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型名称',
  `model_source` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型来源',
  `test_demand_desc` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '测试需求描述',
  `model_type` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型类型',
  `model_function` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型功能',
  `build_unit_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '建设单位名称',
  `build_unit_address` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '建设单位地址',
  `build_unit_leader` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '建设单位负责人',
  `build_unit_contact` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '建设单位负责人联系方式',
  `bt_unit_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '承建单位名称',
  `bt_unit_address` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '承建单位地址',
  `bt_unit_leader` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '承建单位负责人',
  `bt_unit_contact` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '建设单位负责人联系方式',
  `apply_for_type` int(1) NULL DEFAULT NULL COMMENT '申请类型-> 1：文本申请 、 2：系统申请',
  `model_way` int(1) NULL DEFAULT NULL COMMENT '模型方式-> 1：模型测试 、 2：模型评估',
  `apply_for_pdf` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '回填pdf报告路径',
  `apply_for_status` int(1) NULL DEFAULT NULL COMMENT '申请状态-> \r\n1：草稿 、 \r\n2：审批中 、\r\n3：审批通过 、\r\n4：审批打回\r\n5：已完成',
  `approve_status` int(1) NULL DEFAULT NULL COMMENT '审批状态',
  `apply_for_date` date NULL DEFAULT NULL COMMENT '申请日期',
  `user_id` int(8) NULL DEFAULT NULL COMMENT '用户id',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型基础信息表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_base
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_code
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_code`;
CREATE TABLE `qczy_model_code`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `model_base_id` int(8) NULL DEFAULT NULL COMMENT '模型id',
  `code` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '编码',
  `algorithm` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '算法',
  `chinese_info` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '中文信息',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_code
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_configure
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_configure`;
CREATE TABLE `qczy_model_configure`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `model_base_id` int(8) NULL DEFAULT NULL COMMENT '模型基础信息id',
  `model_encap_way` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型封装方式',
  `model_deploy_addr` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型部署位置',
  `model_file_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型文件名称',
  `model_file_size` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型文件大小',
  `model_interface_desc` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型API接口说明（此处为文件路径）',
  `model_port` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型对外暴露端口',
  `model_cuda_version` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型cuda版本',
  `model_drive_version` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型驱动版本',
  `model_case` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型案例（此处为文件路径）',
  `model_scene` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型检查场景',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `test_indic` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '测试指标',
  `test_indic_grid` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '国网企标',
  `model_algorithm_code` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '算法编码',
  `test_indic_map` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL,
  `model_hash_value` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'SHA256校验',
  `model_md5_value` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT 'md5校验',
  `train_sample` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '训练样本',
  `model_class` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '模型识别类别',
  `model_train_code` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型训练代码 （此处为文件路径）',
  `assess_chart` varchar(300) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '评估图表',
  `test_case` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '测试指标',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型工作表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_configure
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_debug_log
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_debug_log`;
CREATE TABLE `qczy_model_debug_log`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `model_base_id` int(8) NULL DEFAULT NULL COMMENT '模型基础信息id',
  `model_address` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '模型地址',
  `request_type` int(8) NULL DEFAULT NULL COMMENT '接口请求方式 - >\r\n1：post\r\n2:  get\r\n3:  put',
  `apply_for_type` int(8) NULL DEFAULT NULL COMMENT '提交方式 - > 1 代表 Json 、2 代表 Excel',
  `test_file_base64` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '测试文件地址',
  `debug_params` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '调试参数',
  `debug_result` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '调试结果',
  `debug_status` int(8) NULL DEFAULT NULL COMMENT '调试是否成功  -> 0: 失败 、 1：成功',
  `debug_time` datetime NULL DEFAULT NULL COMMENT '调试时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '模型调试日志记录表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_debug_log
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_model_mark_info
-- ----------------------------
DROP TABLE IF EXISTS `qczy_model_mark_info`;
CREATE TABLE `qczy_model_mark_info`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `task_id` int(8) NULL DEFAULT NULL COMMENT '任务id',
  `son_id` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '数据集id',
  `file_id` int(8) NULL DEFAULT NULL COMMENT '文件id',
  `mark_info` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '标注信息',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_model_mark_info
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_oper_log
-- ----------------------------
DROP TABLE IF EXISTS `qczy_oper_log`;
CREATE TABLE `qczy_oper_log`  (
  `oper_id` int(11) NOT NULL AUTO_INCREMENT COMMENT '日志主键',
  `title` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '模块标题',
  `business_type` int(11) NULL DEFAULT 0 COMMENT '业务类型（0其它 1新增 2修改 3删除）',
  `method` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '方法名称',
  `request_method` varchar(10) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '请求方式',
  `operator_type` int(11) NULL DEFAULT 0 COMMENT '操作类别（0其它 1后台用户 2手机端用户）',
  `oper_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '操作人员',
  `oper_url` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '请求URL',
  `oper_ip` varchar(128) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '主机地址',
  `oper_location` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '操作地点',
  `oper_param` varchar(2000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '请求参数',
  `json_result` varchar(2000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '返回参数',
  `status` int(11) NULL DEFAULT 0 COMMENT '操作状态（0正常 1异常）',
  `error_msg` varchar(2000) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '' COMMENT '错误消息',
  `oper_time` datetime NULL DEFAULT NULL COMMENT '操作时间',
  `cost_time` bigint(20) NULL DEFAULT 0 COMMENT '消耗时间',
  PRIMARY KEY (`oper_id`) USING BTREE,
  INDEX `idx_sys_oper_log_bt`(`business_type`) USING BTREE,
  INDEX `idx_sys_oper_log_s`(`status`) USING BTREE,
  INDEX `idx_sys_oper_log_ot`(`oper_time`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '操作日志记录' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_oper_log
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_role
-- ----------------------------
DROP TABLE IF EXISTS `qczy_role`;
CREATE TABLE `qczy_role`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '角色id',
  `role_name` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色名称',
  `role_code` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色编码',
  `role_desc` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '角色描述',
  `status` int(11) NULL DEFAULT NULL COMMENT '菜单状态：1->启用、2->禁用',
  `mul_status` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '多人标注权限：1. 发起任务  2. 标注员 3.审核员',
  `is_allow_deletion` int(11) NULL DEFAULT NULL COMMENT '是否允许删除：1-> 不允许 、 2->允许',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `is_deleted` int(11) NOT NULL COMMENT '是否删除：0->正常、1->已删除',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 17 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '角色表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_role
-- ----------------------------
INSERT INTO `qczy_role` VALUES (1, '管理员', 'ADMIN', 'Have All Permissions', 1, '0', 2, '2024-07-26 09:50:28', '2024-08-05 17:53:05', 0);
INSERT INTO `qczy_role` VALUES (8, '标注员', 'ANNOTATER', '标注员角色', 1, '1', 2, '2025-03-04 10:52:36', '2025-03-05 16:44:07', 0);
INSERT INTO `qczy_role` VALUES (11, '审核员', 'AUDITOR', '审核员角色', 1, '2', 1, '2025-03-17 17:17:04', NULL, 0);
INSERT INTO `qczy_role` VALUES (14, '模型评估-申请员', 'MODEL_APPLY_FOR', '厂商角色', 1, '3', 2, '2025-05-29 14:07:17', '2025-05-29 16:12:16', 0);
INSERT INTO `qczy_role` VALUES (15, '模型评估-审核员', 'MODEL_REVIEWER', '模型评估-审核员角色', 1, '3', 2, '2025-05-29 16:13:38', '2025-05-29 16:10:38', 0);
INSERT INTO `qczy_role` VALUES (16, '模型评估-领导', 'LEADER', '领导角色', 1, '3', 2, '2025-05-29 16:14:49', '2025-05-29 16:10:38', 0);

-- ----------------------------
-- Table structure for qczy_role_button
-- ----------------------------
DROP TABLE IF EXISTS `qczy_role_button`;
CREATE TABLE `qczy_role_button`  (
  `role_id` int(11) NOT NULL COMMENT '角色id',
  `button_id` int(11) NOT NULL COMMENT '按钮id'
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_role_button
-- ----------------------------
INSERT INTO `qczy_role_button` VALUES (12, 101);
INSERT INTO `qczy_role_button` VALUES (12, 102);
INSERT INTO `qczy_role_button` VALUES (12, 105);
INSERT INTO `qczy_role_button` VALUES (12, 106);
INSERT INTO `qczy_role_button` VALUES (12, 103);
INSERT INTO `qczy_role_button` VALUES (12, 104);
INSERT INTO `qczy_role_button` VALUES (13, 103);
INSERT INTO `qczy_role_button` VALUES (13, 104);
INSERT INTO `qczy_role_button` VALUES (13, 106);
INSERT INTO `qczy_role_button` VALUES (13, 107);
INSERT INTO `qczy_role_button` VALUES (16, 103);
INSERT INTO `qczy_role_button` VALUES (16, 108);
INSERT INTO `qczy_role_button` VALUES (15, 108);
INSERT INTO `qczy_role_button` VALUES (15, 109);
INSERT INTO `qczy_role_button` VALUES (15, 110);
INSERT INTO `qczy_role_button` VALUES (15, 111);
INSERT INTO `qczy_role_button` VALUES (14, 103);
INSERT INTO `qczy_role_button` VALUES (14, 104);
INSERT INTO `qczy_role_button` VALUES (14, 105);
INSERT INTO `qczy_role_button` VALUES (14, 106);
INSERT INTO `qczy_role_button` VALUES (14, 107);
INSERT INTO `qczy_role_button` VALUES (14, 112);
INSERT INTO `qczy_role_button` VALUES (1, 6);
INSERT INTO `qczy_role_button` VALUES (1, 7);
INSERT INTO `qczy_role_button` VALUES (1, 8);
INSERT INTO `qczy_role_button` VALUES (1, 101);
INSERT INTO `qczy_role_button` VALUES (1, 102);
INSERT INTO `qczy_role_button` VALUES (1, 103);
INSERT INTO `qczy_role_button` VALUES (1, 104);
INSERT INTO `qczy_role_button` VALUES (1, 105);
INSERT INTO `qczy_role_button` VALUES (1, 106);
INSERT INTO `qczy_role_button` VALUES (1, 107);
INSERT INTO `qczy_role_button` VALUES (1, 108);
INSERT INTO `qczy_role_button` VALUES (1, 109);
INSERT INTO `qczy_role_button` VALUES (1, 110);
INSERT INTO `qczy_role_button` VALUES (1, 111);
INSERT INTO `qczy_role_button` VALUES (1, 112);
INSERT INTO `qczy_role_button` VALUES (1, 113);
INSERT INTO `qczy_role_button` VALUES (1, 114);
INSERT INTO `qczy_role_button` VALUES (1, 115);
INSERT INTO `qczy_role_button` VALUES (1, 116);
INSERT INTO `qczy_role_button` VALUES (1, 117);
INSERT INTO `qczy_role_button` VALUES (1, 118);
INSERT INTO `qczy_role_button` VALUES (1, 119);
INSERT INTO `qczy_role_button` VALUES (1, 120);
INSERT INTO `qczy_role_button` VALUES (1, 121);
INSERT INTO `qczy_role_button` VALUES (1, 125);
INSERT INTO `qczy_role_button` VALUES (1, 126);
INSERT INTO `qczy_role_button` VALUES (1, 122);
INSERT INTO `qczy_role_button` VALUES (1, 127);
INSERT INTO `qczy_role_button` VALUES (1, 124);
INSERT INTO `qczy_role_button` VALUES (1, 123);
INSERT INTO `qczy_role_button` VALUES (1, 128);
INSERT INTO `qczy_role_button` VALUES (1, 129);
INSERT INTO `qczy_role_button` VALUES (1, 130);
INSERT INTO `qczy_role_button` VALUES (1, 131);

-- ----------------------------
-- Table structure for qczy_role_menu
-- ----------------------------
DROP TABLE IF EXISTS `qczy_role_menu`;
CREATE TABLE `qczy_role_menu`  (
  `role_id` int(11) NOT NULL COMMENT '角色id',
  `menu_id` int(11) NOT NULL COMMENT '菜单id'
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '角色关联菜单表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_role_menu
-- ----------------------------
INSERT INTO `qczy_role_menu` VALUES (7, 16);
INSERT INTO `qczy_role_menu` VALUES (7, 101);
INSERT INTO `qczy_role_menu` VALUES (7, 102);
INSERT INTO `qczy_role_menu` VALUES (6, 101);
INSERT INTO `qczy_role_menu` VALUES (6, 102);
INSERT INTO `qczy_role_menu` VALUES (6, 16);
INSERT INTO `qczy_role_menu` VALUES (6, 1);
INSERT INTO `qczy_role_menu` VALUES (8, 1);
INSERT INTO `qczy_role_menu` VALUES (8, 9);
INSERT INTO `qczy_role_menu` VALUES (8, 16);
INSERT INTO `qczy_role_menu` VALUES (8, 53);
INSERT INTO `qczy_role_menu` VALUES (13, 2);
INSERT INTO `qczy_role_menu` VALUES (13, 3);
INSERT INTO `qczy_role_menu` VALUES (13, 4);
INSERT INTO `qczy_role_menu` VALUES (13, 5);
INSERT INTO `qczy_role_menu` VALUES (13, 6);
INSERT INTO `qczy_role_menu` VALUES (13, 7);
INSERT INTO `qczy_role_menu` VALUES (13, 8);
INSERT INTO `qczy_role_menu` VALUES (13, 1);
INSERT INTO `qczy_role_menu` VALUES (11, 9);
INSERT INTO `qczy_role_menu` VALUES (11, 16);
INSERT INTO `qczy_role_menu` VALUES (11, 1);
INSERT INTO `qczy_role_menu` VALUES (11, 53);
INSERT INTO `qczy_role_menu` VALUES (12, 101);
INSERT INTO `qczy_role_menu` VALUES (12, 17);
INSERT INTO `qczy_role_menu` VALUES (12, 102);
INSERT INTO `qczy_role_menu` VALUES (12, 18);
INSERT INTO `qczy_role_menu` VALUES (12, 1);
INSERT INTO `qczy_role_menu` VALUES (8, 65);
INSERT INTO `qczy_role_menu` VALUES (11, 65);
INSERT INTO `qczy_role_menu` VALUES (1, 1);
INSERT INTO `qczy_role_menu` VALUES (1, 17);
INSERT INTO `qczy_role_menu` VALUES (1, 18);
INSERT INTO `qczy_role_menu` VALUES (1, 60);
INSERT INTO `qczy_role_menu` VALUES (1, 61);
INSERT INTO `qczy_role_menu` VALUES (1, 1);
INSERT INTO `qczy_role_menu` VALUES (1, 60);
INSERT INTO `qczy_role_menu` VALUES (1, 61);
INSERT INTO `qczy_role_menu` VALUES (1, 1);
INSERT INTO `qczy_role_menu` VALUES (1, 60);
INSERT INTO `qczy_role_menu` VALUES (1, 61);
INSERT INTO `qczy_role_menu` VALUES (1, 2);
INSERT INTO `qczy_role_menu` VALUES (1, 3);
INSERT INTO `qczy_role_menu` VALUES (1, 4);
INSERT INTO `qczy_role_menu` VALUES (1, 5);
INSERT INTO `qczy_role_menu` VALUES (1, 6);
INSERT INTO `qczy_role_menu` VALUES (1, 7);
INSERT INTO `qczy_role_menu` VALUES (1, 8);
INSERT INTO `qczy_role_menu` VALUES (1, 9);
INSERT INTO `qczy_role_menu` VALUES (1, 10);
INSERT INTO `qczy_role_menu` VALUES (1, 11);
INSERT INTO `qczy_role_menu` VALUES (1, 12);
INSERT INTO `qczy_role_menu` VALUES (1, 13);
INSERT INTO `qczy_role_menu` VALUES (1, 14);
INSERT INTO `qczy_role_menu` VALUES (1, 15);
INSERT INTO `qczy_role_menu` VALUES (1, 16);
INSERT INTO `qczy_role_menu` VALUES (1, 19);
INSERT INTO `qczy_role_menu` VALUES (1, 20);
INSERT INTO `qczy_role_menu` VALUES (1, 21);
INSERT INTO `qczy_role_menu` VALUES (1, 22);
INSERT INTO `qczy_role_menu` VALUES (1, 23);
INSERT INTO `qczy_role_menu` VALUES (1, 24);
INSERT INTO `qczy_role_menu` VALUES (1, 25);
INSERT INTO `qczy_role_menu` VALUES (1, 26);
INSERT INTO `qczy_role_menu` VALUES (1, 27);
INSERT INTO `qczy_role_menu` VALUES (1, 28);
INSERT INTO `qczy_role_menu` VALUES (1, 29);
INSERT INTO `qczy_role_menu` VALUES (1, 30);
INSERT INTO `qczy_role_menu` VALUES (1, 31);
INSERT INTO `qczy_role_menu` VALUES (1, 32);
INSERT INTO `qczy_role_menu` VALUES (1, 33);
INSERT INTO `qczy_role_menu` VALUES (1, 34);
INSERT INTO `qczy_role_menu` VALUES (1, 35);
INSERT INTO `qczy_role_menu` VALUES (1, 36);
INSERT INTO `qczy_role_menu` VALUES (1, 37);
INSERT INTO `qczy_role_menu` VALUES (1, 38);
INSERT INTO `qczy_role_menu` VALUES (1, 39);
INSERT INTO `qczy_role_menu` VALUES (1, 40);
INSERT INTO `qczy_role_menu` VALUES (1, 41);
INSERT INTO `qczy_role_menu` VALUES (1, 42);
INSERT INTO `qczy_role_menu` VALUES (1, 43);
INSERT INTO `qczy_role_menu` VALUES (1, 44);
INSERT INTO `qczy_role_menu` VALUES (1, 45);
INSERT INTO `qczy_role_menu` VALUES (1, 46);
INSERT INTO `qczy_role_menu` VALUES (1, 47);
INSERT INTO `qczy_role_menu` VALUES (1, 48);
INSERT INTO `qczy_role_menu` VALUES (1, 49);
INSERT INTO `qczy_role_menu` VALUES (1, 50);
INSERT INTO `qczy_role_menu` VALUES (1, 101);
INSERT INTO `qczy_role_menu` VALUES (1, 102);
INSERT INTO `qczy_role_menu` VALUES (1, 53);
INSERT INTO `qczy_role_menu` VALUES (1, 54);
INSERT INTO `qczy_role_menu` VALUES (1, 55);
INSERT INTO `qczy_role_menu` VALUES (1, 56);
INSERT INTO `qczy_role_menu` VALUES (1, 57);
INSERT INTO `qczy_role_menu` VALUES (1, 58);
INSERT INTO `qczy_role_menu` VALUES (1, 59);
INSERT INTO `qczy_role_menu` VALUES (1, 62);
INSERT INTO `qczy_role_menu` VALUES (1, 63);
INSERT INTO `qczy_role_menu` VALUES (1, 64);
INSERT INTO `qczy_role_menu` VALUES (1, 65);
INSERT INTO `qczy_role_menu` VALUES (14, 1);
INSERT INTO `qczy_role_menu` VALUES (14, 17);
INSERT INTO `qczy_role_menu` VALUES (14, 18);
INSERT INTO `qczy_role_menu` VALUES (14, 60);
INSERT INTO `qczy_role_menu` VALUES (14, 61);
INSERT INTO `qczy_role_menu` VALUES (14, 60);
INSERT INTO `qczy_role_menu` VALUES (14, 1);
INSERT INTO `qczy_role_menu` VALUES (14, 56);
INSERT INTO `qczy_role_menu` VALUES (14, 55);
INSERT INTO `qczy_role_menu` VALUES (15, 1);
INSERT INTO `qczy_role_menu` VALUES (15, 17);
INSERT INTO `qczy_role_menu` VALUES (15, 18);
INSERT INTO `qczy_role_menu` VALUES (15, 60);
INSERT INTO `qczy_role_menu` VALUES (15, 61);
INSERT INTO `qczy_role_menu` VALUES (15, 1);
INSERT INTO `qczy_role_menu` VALUES (15, 60);
INSERT INTO `qczy_role_menu` VALUES (15, 61);
INSERT INTO `qczy_role_menu` VALUES (15, 1);
INSERT INTO `qczy_role_menu` VALUES (15, 60);
INSERT INTO `qczy_role_menu` VALUES (15, 61);
INSERT INTO `qczy_role_menu` VALUES (15, 57);
INSERT INTO `qczy_role_menu` VALUES (15, 55);
INSERT INTO `qczy_role_menu` VALUES (15, 2);
INSERT INTO `qczy_role_menu` VALUES (15, 3);
INSERT INTO `qczy_role_menu` VALUES (15, 4);
INSERT INTO `qczy_role_menu` VALUES (16, 1);
INSERT INTO `qczy_role_menu` VALUES (16, 17);
INSERT INTO `qczy_role_menu` VALUES (16, 18);
INSERT INTO `qczy_role_menu` VALUES (16, 60);
INSERT INTO `qczy_role_menu` VALUES (16, 61);
INSERT INTO `qczy_role_menu` VALUES (16, 1);
INSERT INTO `qczy_role_menu` VALUES (16, 60);
INSERT INTO `qczy_role_menu` VALUES (16, 55);
INSERT INTO `qczy_role_menu` VALUES (16, 56);
INSERT INTO `qczy_role_menu` VALUES (16, 57);
INSERT INTO `qczy_role_menu` VALUES (16, 58);
INSERT INTO `qczy_role_menu` VALUES (16, 59);

-- ----------------------------
-- Table structure for qczy_system_status
-- ----------------------------
DROP TABLE IF EXISTS `qczy_system_status`;
CREATE TABLE `qczy_system_status`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `type` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `gpu_total` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `gpu_used` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `gpu_free` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `gpu_usage` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `cpu_total` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `cpu_used` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `cpu_free` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `cpu_usage` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `mem_total` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `mem_used` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `mem_free` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `mem_usage` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `sys_computer_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_computer_ip` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_user_dir` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_os_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_os_arch` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_fileinfo_total` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_fileinfo_used` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_fileinfo_free` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `sys_fileinfo_usage` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT '',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间-年月日',
  `update_time` datetime NULL DEFAULT NULL COMMENT '时分秒',
  PRIMARY KEY (`id`) USING BTREE,
  INDEX `idx_type_create_time`(`type`, `create_time`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 60 CHARACTER SET = utf8mb4 COLLATE = utf8mb4_general_ci ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_system_status
-- ----------------------------
INSERT INTO `qczy_system_status` VALUES (1, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '437.0', '5657.0', '7.17', '1.706758144E10', '1.465255936E10', '2.41502208E9', '85.85', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78769092608E11', '7.24553035776E11', '34.33', '2025-03-27 18:24:53', '2025-03-27 18:24:53');
INSERT INTO `qczy_system_status` VALUES (2, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '234.0', '5860.0', '3.84', '1.706758144E10', '1.4690230272E10', '2.377351168E9', '86.07', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78769420288E11', '7.24552708096E11', '34.33', '2025-03-27 18:24:56', '2025-03-27 18:24:56');
INSERT INTO `qczy_system_status` VALUES (3, '1', '0.0', '0.0', '0.0', '0.0', '6002.0', '235.0', '5767.0', '3.92', '1.706758144E10', '1.4699466752E10', '2.368114688E9', '86.13', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78770141184E11', '7.245519872E11', '34.33', '2025-03-27 18:25:01', '2025-03-27 18:25:01');
INSERT INTO `qczy_system_status` VALUES (4, '1', '0.0', '0.0', '0.0', '0.0', '6014.0', '264.0', '5750.0', '4.39', '1.706758144E10', '1.4696521728E10', '2.371059712E9', '86.11', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877065728E11', '7.24551471104E11', '34.33', '2025-03-27 18:25:06', '2025-03-27 18:25:06');
INSERT INTO `qczy_system_status` VALUES (5, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '250.0', '5750.0', '4.17', '1.706758144E10', '1.4710407168E10', '2.357174272E9', '86.19', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78771255296E11', '7.24550873088E11', '34.33', '2025-03-27 18:25:11', '2025-03-27 18:25:11');
INSERT INTO `qczy_system_status` VALUES (6, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '280.0', '5720.0', '4.67', '1.706758144E10', '1.4708801536E10', '2.358779904E9', '86.18', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78771648512E11', '7.24550479872E11', '34.33', '2025-03-27 18:25:16', '2025-03-27 18:25:16');
INSERT INTO `qczy_system_status` VALUES (7, '1', '0.0', '0.0', '0.0', '0.0', '6015.0', '202.0', '5813.0', '3.36', '1.706758144E10', '1.47026944E10', '2.36488704E9', '86.14', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78772238336E11', '7.24549890048E11', '34.33', '2025-03-27 18:25:21', '2025-03-27 18:25:21');
INSERT INTO `qczy_system_status` VALUES (8, '1', '0.0', '0.0', '0.0', '0.0', '5999.0', '109.0', '5890.0', '1.82', '1.706758144E10', '1.4700261376E10', '2.367320064E9', '86.13', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.787725824E11', '7.24549545984E11', '34.33', '2025-03-27 18:25:26', '2025-03-27 18:25:26');
INSERT INTO `qczy_system_status` VALUES (9, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '250.0', '5844.0', '4.1', '1.706758144E10', '1.4708310016E10', '2.359271424E9', '86.18', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78773057536E11', '7.24549070848E11', '34.33', '2025-03-27 18:25:31', '2025-03-27 18:25:31');
INSERT INTO `qczy_system_status` VALUES (10, '1', '0.0', '0.0', '0.0', '0.0', '6109.0', '172.0', '5937.0', '2.82', '1.706758144E10', '1.47180544E10', '2.34952704E9', '86.23', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78762932224E11', '7.2455919616E11', '34.33', '2025-03-27 18:25:36', '2025-03-27 18:25:36');
INSERT INTO `qczy_system_status` VALUES (11, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '375.0', '5719.0', '6.15', '1.706758144E10', '1.4717444096E10', '2.350137344E9', '86.23', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78763481088E11', '7.24558647296E11', '34.33', '2025-03-27 18:25:41', '2025-03-27 18:25:41');
INSERT INTO `qczy_system_status` VALUES (12, '1', '0.0', '0.0', '0.0', '0.0', '6031.0', '297.0', '5734.0', '4.92', '1.706758144E10', '1.471739904E10', '2.3501824E9', '86.23', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78764070912E11', '7.24558057472E11', '34.33', '2025-03-27 18:25:46', '2025-03-27 18:25:46');
INSERT INTO `qczy_system_status` VALUES (13, '1', '0.0', '0.0', '0.0', '0.0', '6095.0', '563.0', '5532.0', '9.24', '1.706758144E10', '1.4734073856E10', '2.333507584E9', '86.33', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7876451328E11', '7.24557615104E11', '34.33', '2025-03-27 18:25:51', '2025-03-27 18:25:51');
INSERT INTO `qczy_system_status` VALUES (14, '1', '0.0', '0.0', '0.0', '0.0', '6001.0', '281.0', '5720.0', '4.68', '1.706758144E10', '1.4722629632E10', '2.344951808E9', '86.26', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78765033472E11', '7.24557094912E11', '34.33', '2025-03-27 18:25:56', '2025-03-27 18:25:56');
INSERT INTO `qczy_system_status` VALUES (15, '1', '0.0', '0.0', '0.0', '0.0', '6093.0', '281.0', '5812.0', '4.61', '1.706758144E10', '1.4725296128E10', '2.342285312E9', '86.28', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78765492224E11', '7.2455663616E11', '34.33', '2025-03-27 18:26:01', '2025-03-27 18:26:01');
INSERT INTO `qczy_system_status` VALUES (16, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '358.0', '5642.0', '5.97', '1.706758144E10', '1.4759706624E10', '2.307874816E9', '86.48', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78771091456E11', '7.24551036928E11', '34.33', '2025-03-27 18:26:06', '2025-03-27 18:26:06');
INSERT INTO `qczy_system_status` VALUES (17, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '359.0', '5641.0', '5.98', '1.706758144E10', '1.4761316352E10', '2.306265088E9', '86.49', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78771628032E11', '7.24550500352E11', '34.33', '2025-03-27 18:26:11', '2025-03-27 18:26:11');
INSERT INTO `qczy_system_status` VALUES (18, '1', '0.0', '0.0', '0.0', '0.0', '6093.0', '266.0', '5827.0', '4.37', '1.706758144E10', '1.4759657472E10', '2.307923968E9', '86.48', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78772217856E11', '7.24549910528E11', '34.33', '2025-03-27 18:26:16', '2025-03-27 18:26:16');
INSERT INTO `qczy_system_status` VALUES (19, '1', '0.0', '0.0', '0.0', '0.0', '6093.0', '437.0', '5656.0', '7.17', '1.706758144E10', '1.4873329664E10', '2.194251776E9', '87.14', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877254144E11', '7.24549586944E11', '34.33', '2025-03-27 18:26:21', '2025-03-27 18:26:21');
INSERT INTO `qczy_system_status` VALUES (20, '1', '0.0', '0.0', '0.0', '0.0', '6078.0', '203.0', '5875.0', '3.34', '1.706758144E10', '1.4829101056E10', '2.238480384E9', '86.88', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78773266432E11', '7.24548861952E11', '34.33', '2025-03-27 18:26:26', '2025-03-27 18:26:26');
INSERT INTO `qczy_system_status` VALUES (21, '1', '0.0', '0.0', '0.0', '0.0', '5999.0', '125.0', '5874.0', '2.08', '1.706758144E10', '1.4835642368E10', '2.231939072E9', '86.92', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877374976E11', '7.24548378624E11', '34.33', '2025-03-27 18:26:31', '2025-03-27 18:26:31');
INSERT INTO `qczy_system_status` VALUES (22, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '361.0', '5639.0', '6.02', '1.706758144E10', '1.4794797056E10', '2.272784384E9', '86.68', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877211136E11', '7.24550017024E11', '34.33', '2025-03-27 18:26:36', '2025-03-27 18:26:36');
INSERT INTO `qczy_system_status` VALUES (23, '1', '0.0', '0.0', '0.0', '0.0', '6079.0', '1015.0', '5064.0', '16.7', '1.706758144E10', '1.481234432E10', '2.25523712E9', '86.79', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78772639744E11', '7.2454948864E11', '34.33', '2025-03-27 18:26:41', '2025-03-27 18:26:41');
INSERT INTO `qczy_system_status` VALUES (24, '1', '0.0', '0.0', '0.0', '0.0', '6091.0', '264.0', '5827.0', '4.33', '1.706758144E10', '1.4800347136E10', '2.267234304E9', '86.72', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78772901888E11', '7.24549226496E11', '34.33', '2025-03-27 18:26:46', '2025-03-27 18:26:46');
INSERT INTO `qczy_system_status` VALUES (25, '1', '0.0', '0.0', '0.0', '0.0', '6079.0', '172.0', '5907.0', '2.83', '1.706758144E10', '1.4741848064E10', '2.325733376E9', '86.37', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78773295104E11', '7.2454883328E11', '34.33', '2025-03-27 18:26:51', '2025-03-27 18:26:51');
INSERT INTO `qczy_system_status` VALUES (26, '1', '0.0', '0.0', '0.0', '0.0', '6096.0', '329.0', '5767.0', '5.4', '1.706758144E10', '1.4739423232E10', '2.328158208E9', '86.36', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78773819392E11', '7.24548308992E11', '34.33', '2025-03-27 18:26:56', '2025-03-27 18:26:56');
INSERT INTO `qczy_system_status` VALUES (27, '1', '0.0', '0.0', '0.0', '0.0', '6092.0', '218.0', '5874.0', '3.58', '1.706758144E10', '1.4741327872E10', '2.326253568E9', '86.37', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78774147072E11', '7.24547981312E11', '34.33', '2025-03-27 18:27:01', '2025-03-27 18:27:01');
INSERT INTO `qczy_system_status` VALUES (28, '1', '0.0', '0.0', '0.0', '0.0', '6095.0', '266.0', '5829.0', '4.36', '1.706758144E10', '1.4742126592E10', '2.325454848E9', '86.38', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877467136E11', '7.24547457024E11', '34.33', '2025-03-27 18:27:06', '2025-03-27 18:27:06');
INSERT INTO `qczy_system_status` VALUES (29, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '281.0', '5813.0', '4.61', '1.706758144E10', '1.4823964672E10', '2.243616768E9', '86.85', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78775351296E11', '7.24546777088E11', '34.33', '2025-03-27 18:27:11', '2025-03-27 18:27:11');
INSERT INTO `qczy_system_status` VALUES (30, '1', '0.0', '0.0', '0.0', '0.0', '6113.0', '329.0', '5784.0', '5.38', '1.706758144E10', '1.4867922944E10', '2.199658496E9', '87.11', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78775785472E11', '7.24546342912E11', '34.33', '2025-03-27 18:27:16', '2025-03-27 18:27:16');
INSERT INTO `qczy_system_status` VALUES (31, '1', '0.0', '0.0', '0.0', '0.0', '5998.0', '577.0', '5421.0', '9.62', '1.706758144E10', '1.4801309696E10', '2.266271744E9', '86.72', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78776178688E11', '7.24545949696E11', '34.33', '2025-03-27 18:27:21', '2025-03-27 18:27:21');
INSERT INTO `qczy_system_status` VALUES (32, '1', '0.0', '0.0', '0.0', '0.0', '6095.0', '157.0', '5938.0', '2.58', '1.706758144E10', '1.4787272704E10', '2.280308736E9', '86.64', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877665792E11', '7.24545470464E11', '34.33', '2025-03-27 18:27:26', '2025-03-27 18:27:26');
INSERT INTO `qczy_system_status` VALUES (33, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '297.0', '5703.0', '4.95', '1.706758144E10', '1.4782500864E10', '2.285080576E9', '86.61', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78777055232E11', '7.24545073152E11', '34.33', '2025-03-27 18:27:31', '2025-03-27 18:27:31');
INSERT INTO `qczy_system_status` VALUES (34, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '204.0', '5890.0', '3.35', '1.706758144E10', '1.4810329088E10', '2.257252352E9', '86.77', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78767028224E11', '7.2455510016E11', '34.33', '2025-03-27 18:27:36', '2025-03-27 18:27:36');
INSERT INTO `qczy_system_status` VALUES (35, '1', '0.0', '0.0', '0.0', '0.0', '6079.0', '672.0', '5407.0', '11.05', '1.706758144E10', '1.4866296832E10', '2.201284608E9', '87.1', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78767622144E11', '7.2455450624E11', '34.33', '2025-03-27 18:27:41', '2025-03-27 18:27:41');
INSERT INTO `qczy_system_status` VALUES (36, '1', '0.0', '0.0', '0.0', '0.0', '6079.0', '204.0', '5875.0', '3.36', '1.706758144E10', '1.4872981504E10', '2.194599936E9', '87.14', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78768224256E11', '7.24553904128E11', '34.33', '2025-03-27 18:27:46', '2025-03-27 18:27:46');
INSERT INTO `qczy_system_status` VALUES (37, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '281.0', '5813.0', '4.61', '1.706758144E10', '1.4858162176E10', '2.209419264E9', '87.05', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78768748544E11', '7.2455337984E11', '34.33', '2025-03-27 18:27:51', '2025-03-27 18:27:51');
INSERT INTO `qczy_system_status` VALUES (38, '1', '0.0', '0.0', '0.0', '0.0', '6203.0', '1062.0', '5141.0', '17.12', '1.706758144E10', '1.4880538624E10', '2.187042816E9', '87.19', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7876914176E11', '7.24552986624E11', '34.33', '2025-03-27 18:27:56', '2025-03-27 18:27:56');
INSERT INTO `qczy_system_status` VALUES (39, '1', '0.0', '0.0', '0.0', '0.0', '5999.0', '203.0', '5796.0', '3.38', '1.706758144E10', '1.489174528E10', '2.17583616E9', '87.25', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78769600512E11', '7.24552527872E11', '34.33', '2025-03-27 18:28:01', '2025-03-27 18:28:01');
INSERT INTO `qczy_system_status` VALUES (40, '1', '0.0', '0.0', '0.0', '0.0', '6096.0', '329.0', '5767.0', '5.4', '1.706758144E10', '1.4879793152E10', '2.187788288E9', '87.18', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78769997824E11', '7.2455213056E11', '34.33', '2025-03-27 18:28:06', '2025-03-27 18:28:06');
INSERT INTO `qczy_system_status` VALUES (41, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '297.0', '5797.0', '4.87', '1.706758144E10', '1.4864338944E10', '2.203242496E9', '87.09', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.7877045248E11', '7.24551675904E11', '34.33', '2025-03-27 18:28:11', '2025-03-27 18:28:11');
INSERT INTO `qczy_system_status` VALUES (42, '1', '0.0', '0.0', '0.0', '0.0', '6094.0', '172.0', '5922.0', '2.82', '1.706758144E10', '1.4869508096E10', '2.198073344E9', '87.12', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78775547904E11', '7.2454658048E11', '34.33', '2025-03-27 18:28:16', '2025-03-27 18:28:16');
INSERT INTO `qczy_system_status` VALUES (43, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '719.0', '5281.0', '11.98', '1.706758144E10', '1.5156875264E10', '1.910706176E9', '88.81', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78780925952E11', '7.24541202432E11', '34.33', '2025-03-27 18:28:21', '2025-03-27 18:28:21');
INSERT INTO `qczy_system_status` VALUES (44, '1', '0.0', '0.0', '0.0', '0.0', '6093.0', '249.0', '5844.0', '4.09', '1.706758144E10', '1.5147286528E10', '1.920294912E9', '88.75', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78781405184E11', '7.245407232E11', '34.33', '2025-03-27 18:28:26', '2025-03-27 18:28:26');
INSERT INTO `qczy_system_status` VALUES (45, '1', '0.0', '0.0', '0.0', '0.0', '6002.0', '298.0', '5704.0', '4.97', '1.706758144E10', '1.5137370112E10', '1.930211328E9', '88.69', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78781753344E11', '7.2454037504E11', '34.33', '2025-03-27 18:28:31', '2025-03-27 18:28:31');
INSERT INTO `qczy_system_status` VALUES (46, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '313.0', '5687.0', '5.22', '1.706758144E10', '1.5104270336E10', '1.963311104E9', '88.5', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78782212096E11', '7.24539916288E11', '34.33', '2025-03-27 18:28:36', '2025-03-27 18:28:36');
INSERT INTO `qczy_system_status` VALUES (47, '1', '0.0', '0.0', '0.0', '0.0', '6077.0', '328.0', '5749.0', '5.4', '1.706758144E10', '1.5096455168E10', '1.971126272E9', '88.45', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78782662656E11', '7.24539465728E11', '34.33', '2025-03-27 18:28:41', '2025-03-27 18:28:41');
INSERT INTO `qczy_system_status` VALUES (48, '1', '0.0', '0.0', '0.0', '0.0', '6095.0', '203.0', '5892.0', '3.33', '1.706758144E10', '1.5095660544E10', '1.971920896E9', '88.45', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78783121408E11', '7.24539006976E11', '34.33', '2025-03-27 18:28:46', '2025-03-27 18:28:46');
INSERT INTO `qczy_system_status` VALUES (49, '1', '0.0', '0.0', '0.0', '0.0', '6000.0', '250.0', '5750.0', '4.17', '1.706758144E10', '1.5092424704E10', '1.975156736E9', '88.43', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78783576064E11', '7.2453855232E11', '34.33', '2025-03-27 18:28:51', '2025-03-27 18:28:51');
INSERT INTO `qczy_system_status` VALUES (50, '1', '0.0', '0.0', '0.0', '0.0', '6016.0', '376.0', '5640.0', '6.25', '1.706758144E10', '1.509269504E10', '1.9748864E9', '88.43', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78784034816E11', '7.24538093568E11', '34.33', '2025-03-27 18:28:56', '2025-03-27 18:28:56');
INSERT INTO `qczy_system_status` VALUES (51, '1', '0.0', '0.0', '0.0', '0.0', '6048.0', '890.0', '5158.0', '14.72', '1.706758144E10', '1.5142219776E10', '1.925361664E9', '88.72', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78785017856E11', '7.24537110528E11', '34.33', '2025-03-27 18:29:01', '2025-03-27 18:29:01');
INSERT INTO `qczy_system_status` VALUES (52, '1', '0.0', '0.0', '0.0', '0.0', '6017.0', '423.0', '5594.0', '7.03', '1.706758144E10', '1.5150178304E10', '1.917403136E9', '88.77', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78785431552E11', '7.24536696832E11', '34.33', '2025-03-27 18:29:06', '2025-03-27 18:29:06');
INSERT INTO `qczy_system_status` VALUES (53, '1', '0.0', '0.0', '0.0', '0.0', '6079.0', '218.0', '5861.0', '3.59', '1.706758144E10', '1.5146999808E10', '1.920581632E9', '88.75', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78785693696E11', '7.24536434688E11', '34.33', '2025-03-27 18:29:11', '2025-03-27 18:29:11');
INSERT INTO `qczy_system_status` VALUES (54, '1', '0.0', '0.0', '0.0', '0.0', '6481.0', '857.0', '5624.0', '13.22', '1.706758144E10', '1.5169691648E10', '1.897889792E9', '88.88', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78786217984E11', '7.245359104E11', '34.33', '2025-03-27 18:29:16', '2025-03-27 18:29:16');
INSERT INTO `qczy_system_status` VALUES (55, '1', '0.0', '0.0', '0.0', '0.0', '6002.0', '392.0', '5610.0', '6.53', '1.706758144E10', '1.5126900736E10', '1.940680704E9', '88.63', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.787866112E11', '7.24535517184E11', '34.33', '2025-03-27 18:29:21', '2025-03-27 18:29:21');
INSERT INTO `qczy_system_status` VALUES (56, '1', '0.0', '0.0', '0.0', '0.0', '6049.0', '658.0', '5391.0', '10.88', '1.706758144E10', '1.5092125696E10', '1.975455744E9', '88.43', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78793586688E11', '7.24528541696E11', '34.33', '2025-03-27 18:29:26', '2025-03-27 18:29:26');
INSERT INTO `qczy_system_status` VALUES (57, '1', '0.0', '0.0', '0.0', '0.0', '6002.0', '314.0', '5688.0', '5.23', '1.706758144E10', '1.5015043072E10', '2.052538368E9', '87.97', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78794057728E11', '7.24528070656E11', '34.33', '2025-03-27 18:29:31', '2025-03-27 18:29:31');
INSERT INTO `qczy_system_status` VALUES (58, '1', '0.0', '0.0', '0.0', '0.0', '6093.0', '1313.0', '4780.0', '21.55', '1.706758144E10', '1.4988570624E10', '2.079010816E9', '87.82', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78794582016E11', '7.24527546368E11', '34.33', '2025-03-27 18:29:36', '2025-03-27 18:29:36');
INSERT INTO `qczy_system_status` VALUES (59, '1', '0.0', '0.0', '0.0', '0.0', '6092.0', '499.0', '5593.0', '8.19', '1.706758144E10', '1.4995296256E10', '2.072285184E9', '87.86', 'DESKTOP-LKQ0PN3', '192.168.0.107', 'H:\\青创\\数据标注-模板\\datamark-admin', 'Windows 11', 'amd64', '1.103322128384E12', '3.78784485376E11', '7.24537643008E11', '34.33', '2025-03-27 18:29:41', '2025-03-27 18:29:41');

-- ----------------------------
-- Table structure for qczy_team
-- ----------------------------
DROP TABLE IF EXISTS `qczy_team`;
CREATE TABLE `qczy_team`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `team_name` varchar(200) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '团队名称',
  `team_dec` varchar(500) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '团队描述',
  `creator` int(8) NULL DEFAULT NULL COMMENT '创建者id',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  `team_type` int(1) NULL DEFAULT NULL COMMENT '团队类型-> 1:标注团队 、 2:审核团队',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '团队表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_team
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_team_user
-- ----------------------------
DROP TABLE IF EXISTS `qczy_team_user`;
CREATE TABLE `qczy_team_user`  (
  `id` int(8) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `team_id` int(8) NOT NULL COMMENT '团队id',
  `user_id` int(8) NOT NULL COMMENT '用户id',
  `remark` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '备注',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = ' 团队用户表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_team_user
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_temp_file
-- ----------------------------
DROP TABLE IF EXISTS `qczy_temp_file`;
CREATE TABLE `qczy_temp_file`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT '自增id',
  `fd_name` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '文件名称',
  `fd_type` varchar(300) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件类型：(jpg、png、txt...)',
  `fd_suffix` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件后缀：(jpg、png、txt...)',
  `fd_temp_path` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '文件临时路径',
  `fd_access_path` longtext CHARACTER SET utf8 COLLATE utf8_general_ci NULL COMMENT '文件访问路径',
  `fd_size` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '文件大小',
  `width` int(11) NULL DEFAULT NULL COMMENT '宽',
  `height` int(11) NULL DEFAULT NULL COMMENT '高',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '修改时间',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 1 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '临时文件表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_temp_file
-- ----------------------------

-- ----------------------------
-- Table structure for qczy_user
-- ----------------------------
DROP TABLE IF EXISTS `qczy_user`;
CREATE TABLE `qczy_user`  (
  `id` int(11) NOT NULL AUTO_INCREMENT COMMENT 'id',
  `user_name` varchar(30) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '用户名',
  `password` varchar(100) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '密码',
  `nick_name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '昵称',
  `user_gender` int(11) NULL DEFAULT NULL COMMENT '性别：1->男、2->女',
  `user_email` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '邮箱',
  `user_phone` varchar(11) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '手机号',
  `status` char(1) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL COMMENT '状态：0->正常、1->冻结',
  `user_roles` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '角色id',
  `dept_ids` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '部门id',
  `is_allow_deletion` int(11) NOT NULL COMMENT '是否允许删除：1-> 不允许 、 2->允许',
  `create_time` datetime NULL DEFAULT NULL COMMENT '创建时间',
  `update_time` datetime NULL DEFAULT NULL COMMENT '更新时间',
  `is_deleted` int(11) NOT NULL COMMENT '是否删除：0->正常、1->已删除',
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 26 CHARACTER SET = utf8 COLLATE = utf8_general_ci COMMENT = '用户表' ROW_FORMAT = DYNAMIC;

-- ----------------------------
-- Records of qczy_user
-- ----------------------------
INSERT INTO `qczy_user` VALUES (1, 'admin', '$2a$10$Q4rEHfSXZPHzJuIae.RQl.zdk.nM2CaQ6BMOtR6/HEBIZdy4D2bdK', '管理员', 1, '1849499176@qq.com', '18568572362', '1', '1', '1', 1, '2024-08-06 10:55:15', '2025-03-31 15:30:43', 0);

SET FOREIGN_KEY_CHECKS = 1;
