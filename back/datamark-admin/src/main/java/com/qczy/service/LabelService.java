package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.request.AddDataSetLabelRequest;
import com.qczy.model.request.DeleteDataSetLabelRequest;
import com.qczy.model.request.LabelEntityRequest;
import com.qczy.model.response.DataSetLabelResponse;
import com.qczy.model.response.GroupLabelResponse;
import org.jpedal.parser.shape.S;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 15:19
 * @Description:
 */
public interface LabelService {
    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public LabelEntity selectLabelById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param label 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<LabelEntity> selectLabelList(Page<LabelEntity> pageParam, Integer labelGroupId, LabelEntity label);

    /**
     * 新增【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    public int insertLabel(LabelEntity label, LabelGroupEntity entity);

    /**
     * 修改【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    public int updateLabel(LabelEntity label, LabelGroupEntity entity);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteLabelByIds(int[] ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteLabelById(Integer id);


    List<GroupLabelResponse> selectGroupLabel();

    int addDataSetAndLabel(AddDataSetLabelRequest request);

    List<DataSetLabelResponse> selectDataSetLabel(String sonId);

    int deleteDataSetLabel(DeleteDataSetLabelRequest request);

    int addSaveLabel(LabelEntityRequest label);

    int updateDataLabel(LabelEntityRequest label);

    IPage<GroupLabelResponse> selectGroupLabelPage(Page<GroupLabelResponse> pageParam);

    IPage<DataSetLabelResponse> selectDataSetLabelPage(Page<DataSetLabelResponse> pageParam, String sonId, String labelName);


    // 判断标签编码是否存在
    boolean isisExistLabelonlyId(Integer groupId, String onlyId, Integer labelId);

    // 判断当前标签名称是否存在
    boolean isExistLabelName(Integer groupId, String labelName, Integer labelId);

    // 根据数据集id判断标签名称是否存在
    boolean isExistLabelName1(Integer labelId, String sonId, String labelName);


    boolean isExistEnglishLabelName(Integer labelGroupId, String englishLabelName, Integer labelId, LabelGroupEntity labelGroup);

    boolean isExistEnglishLabelName1(Integer labelId, String sonId, String englishLabelName);

    int topUpLabel(Integer labelId,String sonId);


    // 新增多个标签 （绑定数据集）
    int deleteDataSetLabelRequest(DeleteDataSetLabelRequest deleteDataSetLabelRequest);


}
