package com.qczy.model.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.jpedal.parser.shape.S;

import javax.validation.constraints.NotNull;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2025/2/24 9:53
 * @Description:
 */
@Data
@TableName("qczy_team_user")
public class TeamUserEntity {

    //自增id
    @TableId(type = IdType.AUTO)
    private Integer Id;

    /**
     * 团队id
     */
    @NotNull(message="[团队id]不能为空")
    @ApiModelProperty("团队id")
    private Integer teamId;
    /**
     * 用户id
     */
    @NotNull(message="[用户id]不能为空")
    @ApiModelProperty("用户id")
    private Integer userId;


    /**
     *  备注
     */
    private String remark;


}
