package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.RoleEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.UserRequest;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0n
 * @Date: 2024/7/26 9:58
 * @Description:
 */
public interface RoleMapper extends BaseMapper<RoleEntity> {


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public RoleEntity selectRoleById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<RoleEntity> selectRoleList(Page<RoleEntity> pageParam, @Param("req") RoleEntity request);

    /**
     * 新增【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    public int insertRole(RoleEntity role);

    /**
     * 修改【请填写功能名称】
     *
     * @param role 【请填写功能名称】
     * @return 结果
     */
    public int updateRole(RoleEntity role);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteRoleById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteRoleByIds(String[] ids);

}
