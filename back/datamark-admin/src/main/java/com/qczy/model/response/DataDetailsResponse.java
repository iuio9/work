package com.qczy.model.response;

import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/21 11:22
 * @Description:
 */
@Data
public class DataDetailsResponse {

    // 自增id
    private Integer id;

    // 数据集id
    private String sonId;

    // 版本
    private Integer version;

    // 文件id
    private Integer fileId;

    // 标注文件id
    private Integer markFileId;

    // 图片路径
    private String imgPath;

    // 图片源路径
    private String previewImgPath;

    // 标注状态 (0-> 未标注、1-> 已标注)
    private String isMark;

    // 标签
    private String labels;

    // 此数据是否有效 无效
    private Integer isInvalid;



    // 标注信息
    private String markInfo;
    // label标注信息
    private String labelMarkInfo;



    // 图片长
    private Integer width;

    // 图片宽
    private Integer height;



    private Integer operateWidth;

    private Integer operateHeight;

    // 文件名称
    private String fileName;

    // 返回意见
    private String notPassMessage;


}
