package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;


/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/12 9:45
 * @Description:
 */

@Data
@TableName("qczy_many_file")
public class ManyFileEntity {

    /**
     * 自增id
     */
    @ApiModelProperty("自增id")
    private Integer id;

    /**
     * 任务id
     */
    @ApiModelProperty("任务id")
    private Integer taskId;


    /**
     * 标注人员id
     */
    @ApiModelProperty("标注人员id")
    private Integer userId;

    /**
     * 审核人员id
     */
    @ApiModelProperty("审核人员id")
    private Integer auditUserId;


    /**
     * 文件id
     */
    @ApiModelProperty("文件id")
    private Integer fileId;

    /**
     * 审核是否通过
     * 0：未审核
     * 1：审核通过
     * 2：审核未通过
     */
    @ApiModelProperty("审核是否通过")
    private Integer isApprove;

    /**
     * 未通过消息
     */
    @ApiModelProperty("未通过消息")
    private String notPassMessage;


    public ManyFileEntity(Integer taskId, Integer userId, Integer fileId, Integer isApprove) {
        this.taskId = taskId;
        this.userId = userId;
        this.fileId = fileId;
        this.isApprove = isApprove;
    }

    public ManyFileEntity() {
    }

}
