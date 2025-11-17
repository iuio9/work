package com.qczy.common.markInfo;

import com.google.gson.JsonArray;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/11/29 10:35
 * @Description:
 */
@Data
public class FileMarkLog {

    // 唯一id
    private Integer id;
    // 文件信息
    private String fileName;
    // 文件宽
    private double width;
    // 文件高
    private double height;
    // 标注类型  矩形
    private String shape;
    // bbox`：对象的边界框，格式为`[x, y, width, height]`。
    private JsonArray bbox;
    // 多边形-点位
    private JsonArray segmentation;
    // 圆形 - 点位
    private String meta;
    // 标签
    private String labelName;


}
