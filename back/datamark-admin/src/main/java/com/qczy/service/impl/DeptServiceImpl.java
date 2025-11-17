package com.qczy.service.impl;

import cn.hutool.core.convert.Convert;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.DeptMapper;
import com.qczy.mapper.UserMapper;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.model.request.DeleteRequest;
import com.qczy.model.response.DeptUserResponse;
import com.qczy.model.response.UserListResponse;
import com.qczy.service.DeptService;
import org.apache.catalina.User;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import static java.lang.Integer.parseInt;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/6 14:54
 * @Description:
 */
@Service
public class DeptServiceImpl extends ServiceImpl<DeptMapper, DeptEntity> implements DeptService {

    @Autowired
    private DeptMapper deptMapper;

    @Autowired
    private UserMapper userMapper;

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    @Override
    public DeptEntity selectDeptById(Integer id) {
        return deptMapper.selectDeptById(id);
    }

    /**
     * 查询【请填写功能名称】列表
     *
     * @param dept 【请填写功能名称】
     * @return 【请填写功能名称】
     */
    @Override
    public IPage<DeptEntity> selectDeptList(Page<DeptEntity> pageParam, DeptEntity dept) {
        return deptMapper.selectDeptList(pageParam, dept);
    }

    /**
     * 新增【请填写功能名称】
     *
     * @param dept 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int insertDept(DeptEntity dept) {
        dept.setCreateTime(new Date());
        dept.setIsAllowDeletion(SystemConstant.YES_DISABLE_DATA);
        return deptMapper.insertDept(dept);
    }

    /**
     * 修改【请填写功能名称】
     *
     * @param dept 【请填写功能名称】
     * @return 结果
     */
    @Override
    public int updateDept(DeptEntity dept) {
        dept.setUpdateTime(new Date());
        return deptMapper.updateDept(dept);
    }

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteDeptByIds(int[] ids) {
        return deptMapper.deleteDeptByIds(ids);
    }

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    @Override
    public int deleteDeptById(Integer id) {
        return deptMapper.deleteDeptById(id);
    }


    @Override
    public List<DeptEntity> getDeptSelect() {
        return deptMapper.selectList(
                new LambdaQueryWrapper<DeptEntity>()
                        .eq(DeptEntity::getStatus, SystemConstant.SYSTEM_NO_FREEZE)
        );
    }

    @Override
    public int getDeptAndUserCount(int[] ids) {
        return userMapper.selectCount(
                new LambdaQueryWrapper<UserEntity>()
                        .like(UserEntity::getDeptIds, StringUtils.join(ids, ','))
                        .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
        );
    }

    @Override
    public List<DeptUserResponse> getDeptByUserList() {
        List<DeptUserResponse> data = new ArrayList<>();

        List<DeptEntity> deptEntityList = deptMapper.selectList(
                new LambdaQueryWrapper<DeptEntity>()
                        .orderByAsc(DeptEntity::getSort));

        if (CollectionUtils.isEmpty(deptEntityList)) {
            return null;
        }

        for (DeptEntity deptEntity : deptEntityList) {
            DeptUserResponse deptUserResponse = new DeptUserResponse();
            deptUserResponse.setDeptId(deptEntity.getId());
            deptUserResponse.setDeptName(deptEntity.getDeptName());
            deptUserResponse.setSort(deptEntity.getSort());

            List<UserEntity> userData = userMapper.selectList(
                    new LambdaQueryWrapper<UserEntity>()
                            .eq(UserEntity::getStatus, SystemConstant.SYSTEM_NO_FREEZE)
                            .eq(UserEntity::getIsDeleted, SystemConstant.SYSTEM_NO_DISABLE)
            );
            List<UserListResponse> userList = getUserListResponses(deptEntity, userData);
            deptUserResponse.setUserList(userList);
            data.add(deptUserResponse);
        }
        return data;
    }

    private static List<UserListResponse> getUserListResponses(DeptEntity deptEntity, List<UserEntity> userData) {
        List<UserListResponse> userList = new ArrayList<>();
        for (UserEntity userEntity : userData) {
            String[] deptIds = userEntity.getDeptIds().split(",");
            for (String deptId : deptIds) {
                if (deptEntity.getId() == Integer.parseInt(deptId)) {
                    UserListResponse userListResponse = new UserListResponse();
                    userListResponse.setUserId(userEntity.getId());
                    userListResponse.setUserName(userEntity.getUserName());
                    userListResponse.setNickName(userEntity.getNickName());
                    userList.add(userListResponse);
                }
            }
        }
        return userList;
    }
}

