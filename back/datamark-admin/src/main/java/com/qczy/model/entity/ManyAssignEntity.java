package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/27 9:44
 * @Description:
 */
@Data
@TableName("qczy_many_assign")
public class ManyAssignEntity {

    /**
     * 自增id
     */
    @ApiModelProperty("自增id")
    private Integer id;

    /**
     * 多人标注任务id
     */
    @ApiModelProperty("多人标注任务id")
    private Integer manyMarkId;

    /**
     * 数据集id
     */
    @ApiModelProperty("数据集id")
    private String sonId;

    /**
     * 用户id
     */
    @ApiModelProperty("用户id")
    private Integer userId;

    /**
     * 分配文件id集合
     */
    @ApiModelProperty("分配文件id集合")
    private String assignFileIds;

    /**
     * 已完成标注数量
     */
    @ApiModelProperty("已完成标注数量")
    private Integer yesMark;

    /**
     * 已完成标注数量
     */
    @ApiModelProperty("未完成标注数量")
    private Integer noMark;

    /**
     * 完成进度
     */
    @ApiModelProperty("完成进度")
    private String progress;

    /**
     * 用户状态
     * 1. 未开始
     * 2. 标注中
     * 3. 已完成
     * 4. 已结束
     * 5. 转交
     * 6. 验收打回
     * 7. 已提交
     */
    @ApiModelProperty("用户状态")
    private Integer userState;

    /**
     * 创建时间
     */
    @ApiModelProperty("创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;

    /**
     * 此数据是否失效
     */
    @ApiModelProperty("此数据是否失效")
    private Integer isLose;

    /**
     *  此任务是否提交
     */
    @ApiModelProperty("此任务是否提交")
    private Integer isSubmit;

}
