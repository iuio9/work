package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/22 14:50
 * @Description:
 */
@Data
@TableName("qczy_data_son_label")
public class DataSonLabelEntity {

    // id
    private Integer id;

    // 数据集id
    private String sonId;

    // 标签id
    private Integer labelId;

    // 标签数量
    private Integer labelCount;


    // 前端额外传入的字段（这个字段没用）
    @TableField(exist = false)
    private Integer index;
}
