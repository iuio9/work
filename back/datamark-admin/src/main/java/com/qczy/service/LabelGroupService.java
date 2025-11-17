package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.request.AssocDataSetRequest;
import com.qczy.model.request.CopyLabelGroupRequest;
import com.qczy.model.request.FileIdsRequest;
import com.qczy.model.response.LabelGroupAndLabelResponse;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 14:45
 * @Description:
 */
public interface LabelGroupService extends IService<LabelGroupEntity> {
    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public LabelGroupEntity selectLabelGroupById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param labelGroup 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<LabelGroupEntity> selectLabelGroupList(Page<LabelGroupEntity> pageParam, LabelGroupEntity labelGroup);

    /**
     * 新增【请填写功能名称】
     *
     * @param labelGroup 【请填写功能名称】
     * @return 结果
     */
    public int insertLabelGroup(LabelGroupEntity labelGroup);

    /**
     * 修改【请填写功能名称】
     *
     * @param labelGroup 【请填写功能名称】
     * @return 结果
     */
    public int updateLabelGroup(LabelGroupEntity labelGroup);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteLabelGroupByIds(int[] ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteLabelGroupById(Integer id);

    List<LabelGroupAndLabelResponse> selectLabelList();

    List<Integer> getSonIdByLabelGroupIds(String sonId);

    // 根据标签组id 绑定数据集  （数据集跟标签组进行关联）
    void addDataSonAndLabelGroup(String sonId, String groupIds);
    // 根据标签id 绑定数据集  （数据集跟标签进行关联）
    void addDataSonAndLabelIds(String sonId, String tagIds);

    // 复制标签组
    int copyLabelGroup(CopyLabelGroupRequest request);

    // 关联数据集
    int assocDataSet(AssocDataSetRequest request);

    //查询当前数据集的标签状态 （包含几个标签组，几个自定义标签）
    String getDataSonLabelStatus(String sonId);

    // 导入标签
    int importLabel(FileIdsRequest request);

    // 判断当前标签名称是否存在
    boolean isExistLabelGroupName(Integer groupId, String groupName);

    boolean isExistLabelEnglishGroupName(Integer groupId, String englishGroupName);



}
