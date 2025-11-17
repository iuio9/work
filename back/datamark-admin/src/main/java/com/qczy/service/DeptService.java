package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.DeptUserResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/6 14:53
 * @Description:
 */
public interface DeptService {

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public DeptEntity selectDeptById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param dept 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<DeptEntity> selectDeptList(Page<DeptEntity> pageParam, DeptEntity dept);

    /**
     * 新增【请填写功能名称】
     *
     * @param dept 【请填写功能名称】
     * @return 结果
     */
    public int insertDept(DeptEntity dept);

    /**
     * 修改【请填写功能名称】
     *
     * @param dept 【请填写功能名称】
     * @return 结果
     */
    public int updateDept(DeptEntity dept);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteDeptByIds(int [] ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteDeptById(Integer id);


    List<DeptEntity> getDeptSelect();

    int getDeptAndUserCount(int[] ids);

    List<DeptUserResponse> getDeptByUserList();
}
