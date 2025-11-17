package com.qczy.model.request;

import com.qczy.model.entity.DataSonEntity;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.io.Serializable;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 21:53
 * @Description:
 */
@EqualsAndHashCode(callSuper = true)
@Data
public class DataSonEntityRequest extends DataSonEntity implements Serializable {

    /**
     * 数据集组名称
     */
    @ApiModelProperty(value = "数据集组名称")
    private String groupName;

    /**
     * 数据级（类型）id
     */
    @ApiModelProperty(value = "数据级（类型）id")
    private Integer dataTypeId;

    /**
     * 判断是否有标注信息
     **/
    @ApiModelProperty(value = "判断是否有标注信息")
    private Integer isMarkInfo;


    /**
     * 数据标注状态 - > 0 无标注信息 1 有标注信息
     **/
    @ApiModelProperty(value = "数据标注状态 - > 0 无标注信息 1 有标注信息")
    private Integer markStatus;


    /**
     * 导入方式
     **/
    @ApiModelProperty(value = "导入方式 ->  0 上传图片 1上传压缩包")
    private Integer importMode;

    /**
     * 导入方式
     **/
    @ApiModelProperty(value = "标签组id")
    private String groupIds;


    /**
     *  旧的图片id
     */
    private String oldFileIds;

    /**
     *  标签类型   group：标签组 、 single：标签
     */
    private String tagSelectionMode;

    /**
     *  标签集合
     */
    private String tagIds;

}
