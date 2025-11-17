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
 * @Date: 2025/2/26 16:24
 * @Description:
 */
@Data
@TableName("qczy_many_mark")
public class ManyMarkEntity {


    /**
     * 自增id
     */
    @ApiModelProperty("自增id")
    private Integer id;

    /**
     *  任务名称
     */
    @ApiModelProperty("任务名称")
    private String taskName;

    /**
     *  数据集id
     */
    @ApiModelProperty("数据集id")
    private String sonId;

    /**
     *  团队id
     */
    @ApiModelProperty("团队id")
    private Integer teamId;

    /**
     *  团队id
     */
    @ApiModelProperty("审核团队id")
    private Integer auditTeamId;

    /**
     1. 任务分配中
     2. 未开始
     3. 标注中
     4. 标注已完成
     5. 待审核
     6. 审核中
     7. 审核完成
     8. 任务结束
     */
    @ApiModelProperty("任务状态")
    private Integer taskState;

    /**
     *  用户id
     */
    @ApiModelProperty("用户id")
    private Integer userId;

    /**
     * 创建时间
     */
    @ApiModelProperty("创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date createTime;
    /**
     * 修改时间
     */
    @ApiModelProperty("修改时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    private Date updateTime;

}
