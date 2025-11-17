package com.qczy.model.request;

import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/12/30 14:37
 * @Description:
 */
@Data
public class AssocDataSetRequest {


    @ApiModelProperty(value = "数据集list")
    private List<String> dataSetIdList;


    @ApiModelProperty(value = "标签组id")
    private Integer labelGroupId;


}
