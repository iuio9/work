package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.io.Serializable;
import java.util.Date;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/3/17 15:51
 * @Description:
 */
@TableName("qczy_many_audit")
@Data
public class ManyAuditEntity implements Serializable {


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
     *  标注人员id
     */
    @ApiModelProperty("标注人员id")
    private Integer markUserId;

    /**
     * 分配文件id集合
     */
    @ApiModelProperty("分配文件id集合")
    private String auditFileIds;

    /**
     * 已审核数量
     */
    @ApiModelProperty("已审核数量")
    private Integer yesExamine;

    /**
     * 未审核数量
     */
    @ApiModelProperty("未审核数量")
    private Integer noExamine;

    /**
     *  进度
     */
    private Integer progress;

    /**
     *  审核状态
     *  1. 待审核
     *  2. 审核中
     *  3. 审核完成
     *  4. 已提交
     *  5. 已驳回
     *  6. 审核结束
     *  7. 待审核 --- TODO 这里指的是打回任务的时候用
     */
    @ApiModelProperty("审核状态")
    private Integer auditState;


    /**
     *  创建时间
     */
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
