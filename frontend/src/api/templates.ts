import type { AnyTemplateItem } from "../types"
import { api } from "./client"
import type { FillTemplateRequest, ScoreToTemplateRequest, TemplateToXmlRequest } from "./client"

export async function scoreToTemplate(xml: string) {
    const template = await api.scoreToTemplate({ xml: xml } as ScoreToTemplateRequest)
    return template.template
}

export async function fillTemplate(template: AnyTemplateItem[]) {
    const score = await api.fillTemplate({ template: template } as FillTemplateRequest)
    return score.xml
}

export async function templateToXml(template: AnyTemplateItem[]) {
    const score = await api.templateToXml({ template: template } as TemplateToXmlRequest)
    return score.xml
}
