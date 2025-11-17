package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.response.DataSetLabelResponse;
import com.qczy.model.response.GroupLabelResponse;
import org.apache.ibatis.annotations.Param;
import org.springframework.security.core.parameters.P;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 14:42
 * @Description:
 */
public interface LabelGroupMapper extends BaseMapper<LabelGroupEntity> {

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
    IPage<LabelGroupEntity> selectLabelGroupList(Page<LabelGroupEntity> pageParam, @Param("req") LabelGroupEntity labelGroup);

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
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteLabelGroupById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteLabelGroupByIds(int [] ids);

    IPage<GroupLabelResponse> selectGroupLabelPage(Page<GroupLabelResponse> pageParam);

    IPage<DataSetLabelResponse> selectDataSetLabelPage(Page<DataSetLabelResponse> pageParam,
                                                       @Param("sonId") String sonId,
                                                       @Param("labelName") String labelName);
}
