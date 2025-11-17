import { transformRecordToOption } from '@/utils/common';

export const enableStatusRecord: Record<Api.Common.EnableStatus, App.I18n.I18nKey> = {
  '1': 'page.manage.common.status.enable',
  '2': 'page.manage.common.status.disable'
};

export const enableStatusRecord1: Record<Api.Common.EnableStatus, App.I18n.I18nKey> = {
  '0': "成功",
  '1': "失败"
};

export const enableStatusRecord2: Record<Api.Common.EnableStatus, App.I18n.I18nKey> = {
  '0': "失败",
  '1': "成功",
};

export const enableStatusOptions = transformRecordToOption(enableStatusRecord);
export const enableStatusOptions1 = transformRecordToOption(enableStatusRecord2);

export const userGenderRecord: Record<Api.SystemManage.UserGender, App.I18n.I18nKey> = {
  '1': 'page.manage.user.gender.male',
  '2': 'page.manage.user.gender.female'
};

export const userGenderOptions = transformRecordToOption(userGenderRecord);

export const menuTypeRecord: Record<Api.SystemManage.MenuType, App.I18n.I18nKey> = {
  '1': 'page.manage.menu.type.directory',
  '2': 'page.manage.menu.type.menu'
};

export const menuTypeOptions = transformRecordToOption(menuTypeRecord);

export const menuIconTypeRecord: Record<Api.SystemManage.IconType, App.I18n.I18nKey> = {
  '1': 'page.manage.menu.iconType.iconify',
  '2': 'page.manage.menu.iconType.local'
};

export const menuIconTypeOptions = transformRecordToOption(menuIconTypeRecord);

// 是否使用
export const yesOrNoRecord: Record<string, string> = {
  '0': '是',
  '1': '否'
};

export const yesOrNoOptions = transformRecordToOption(yesOrNoRecord);
