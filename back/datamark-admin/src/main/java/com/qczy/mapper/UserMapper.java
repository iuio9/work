package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.UserRequest;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/7/15 15:46
 * @Description:
 */
public interface UserMapper extends BaseMapper<UserEntity> {

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public UserEntity selectUserById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param pageParam 分页信息，【请填写功能名称】
     * @param request   查询参数，【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<UserEntity> selectUserList(Page<UserEntity> pageParam, @Param("req") UserRequest request);

    /**
     * 新增【请填写功能名称】
     *
     * @param User 【请填写功能名称】
     * @return 结果
     */
    public int insertUser(UserEntity User);

    /**
     * 修改【请填写功能名称】
     *
     * @param User 【请填写功能名称】
     * @return 结果
     */
    public int updateUser(UserEntity User);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteUserById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteUserByIds(String[] ids);

}
