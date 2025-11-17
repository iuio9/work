package com.qczy.model.response;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.TableField;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.qczy.model.entity.DictDataEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import org.springframework.format.annotation.DateTimeFormat;

import javax.validation.constraints.NotNull;
import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/9/24 13:21
 * @Description:
 */
@Data
public class DictSetTypeResponse {
    // id
    @ApiModelProperty("id")
    private Integer id;

    // 父id
    /**
     * 父id
     */
    @ApiModelProperty(value = "父id")
    private Integer parentId;

    // 数据集类型
    @ApiModelProperty("数据集类型")
    private String dictLabel;

    // 数量
    @ApiModelProperty("数量")
    private Integer number;

    // 备注
    @ApiModelProperty("备注")
    private String remark;

    @ApiModelProperty(value = "创建时间")
    @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "GMT+8")
    @TableField(fill = FieldFill.INSERT)
    private Date createTime;

    // 文件总数量
    private Long fileSumCount;

    // 当前节点总数量
    private Long nodeSumCount;

    /**
     * 子类
     */
    @TableField(exist = false)
    private List<DictSetTypeResponse> children;

}
