package com.qczy.model.response;

import lombok.Data;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/8 11:23
 * @Description:
 */
@Data
public class DictDataTreeResponse {

    /**
     *  字典标签
     */
    private Integer id;

    /**
     *  字典类型id
     */
    private Integer typeId;

    /**
     *  父id
     */
    private Integer parentId;

    /**
     *  字典标签
     */
    private String dictLabel;


    /**
     *  字典健值
     */
    private String dictValue;

    /**
     *  排序
     */
    private Integer dictSort;

    /**
     *  状态
     */
    private String status;

    /**
     *  备注
     */
    private String remark;


    /**
     * 子类
     */
    private List<DictDataTreeResponse> children;





}
