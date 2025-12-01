package com.qczy.distillation.model.entity;

import com.baomidou.mybatisplus.annotation.*;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * LoRA配置预设实体类
 *
 * 对应数据库表：md_lora_preset
 * 功能：存储LoRA（Low-Rank Adaptation）配置预设
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Data
@TableName("md_lora_preset")
@ApiModel(value = "LoRA配置预设实体", description = "存储LoRA配置预设")
public class MdLoraPresetEntity {

    /**
     * 主键ID（自增）
     */
    @TableId(value = "id", type = IdType.AUTO)
    @ApiModelProperty(value = "主键ID")
    private Long id;

    /**
     * 预设名称
     */
    @TableField("preset_name")
    @ApiModelProperty(value = "预设名称")
    private String presetName;

    /**
     * 预设描述
     */
    @TableField("preset_desc")
    @ApiModelProperty(value = "预设描述")
    private String presetDesc;

    // ========== LoRA参数 ==========

    /**
     * LoRA秩
     */
    @TableField("lora_rank")
    @ApiModelProperty(value = "LoRA秩")
    private Integer loraRank;

    /**
     * LoRA alpha参数
     */
    @TableField("lora_alpha")
    @ApiModelProperty(value = "LoRA alpha参数")
    private Integer loraAlpha;

    /**
     * LoRA dropout率
     */
    @TableField("lora_dropout")
    @ApiModelProperty(value = "LoRA dropout率")
    private BigDecimal loraDropout;

    /**
     * 目标模块（JSON数组）
     */
    @TableField("target_modules")
    @ApiModelProperty(value = "目标模块（JSON数组）")
    private String targetModules;

    // ========== 系统字段 ==========

    /**
     * 创建人
     */
    @TableField("create_by")
    @ApiModelProperty(value = "创建人")
    private String createBy;

    /**
     * 创建时间
     */
    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @ApiModelProperty(value = "创建时间")
    private LocalDateTime createTime;

    /**
     * 更新时间
     */
    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @ApiModelProperty(value = "更新时间")
    private LocalDateTime updateTime;

    /**
     * 删除标志（0-未删除，1-已删除）
     */
    @TableField("del_flag")
    @ApiModelProperty(value = "删除标志（0-未删除，1-已删除）")
    private Integer delFlag;
}
